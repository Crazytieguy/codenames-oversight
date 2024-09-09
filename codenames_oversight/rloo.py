import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import Dataset
from peft import PeftModel, PeftModelForCausalLM  # type: ignore
from pydantic import NonNegativeFloat, NonNegativeInt
from toolz.itertoolz import partition_all
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, set_seed
from trl.trainer.rloo_config import RLOOConfig

from .evaluate_clue import evaluate_clue
from .iterative_sft_trainer import IterativeSFTTrainer
from .models import Clue, ClueCritiques, Critique, Game
from .oversight import (
    NeglectLastNOverSeer,
    NegligentBiasedBaseOverSeer,
    NegligentBiasedJudgeOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
    RobustJudgeOverSeer,
    RobustOverSeer,
    approximate_calibrate_p,
)
from .rloo_trainer import RLOOTrainer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


DATASET_FILE: str
MODEL_DIR: str
OUTPUT_DIR: str
CRITIQUE_MODEL_DIR: Optional[str]
BASE_MODEL: str
LEARNING_RATE: float
CRITIQUER_LEARNING_RATE: float
BATCH_SIZE: int
KL_COEFF: float
PPO_EPOCHS: int
ADVERSARIAL_ALPHA: float
RLOO_K: int


@app.callback()
def set_params(
    dataset_file: str,
    model_dir: str,
    output_dir: str,
    critique_model_dir: Optional[str] = None,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    learning_rate: float = 2e-5,
    critiquer_learning_rate: float = 4e-5,
    batch_size: int = 448,
    kl_coeff: float = 0.06,
    ppo_epochs: int = 4,
    adversarial_alpha: float = 0.0,
    rloo_k: int = 4,
):
    global DATASET_FILE
    global MODEL_DIR
    global OUTPUT_DIR
    global CRITIQUE_MODEL_DIR
    global BASE_MODEL
    global LEARNING_RATE
    global CRITIQUER_LEARNING_RATE
    global BATCH_SIZE
    global KL_COEFF
    global PPO_EPOCHS
    global ADVERSARIAL_ALPHA
    global RLOO_K
    DATASET_FILE = dataset_file
    MODEL_DIR = model_dir
    OUTPUT_DIR = output_dir
    BASE_MODEL = base_model
    CRITIQUE_MODEL_DIR = critique_model_dir
    LEARNING_RATE = learning_rate
    CRITIQUER_LEARNING_RATE = critiquer_learning_rate
    BATCH_SIZE = batch_size
    KL_COEFF = kl_coeff
    PPO_EPOCHS = ppo_epochs
    ADVERSARIAL_ALPHA = adversarial_alpha
    RLOO_K = rloo_k


def main(overseer: OverSeer):
    set_seed(0)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, add_eos_token=False, padding_side="left"
    )  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_game_dataset(DATASET_FILE, tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModelForCausalLM.from_pretrained(
        base_model, MODEL_DIR, is_trainable=False, adapter_name="ref"
    )
    model.load_adapter(MODEL_DIR, "cluer", is_trainable=True)

    if CRITIQUE_MODEL_DIR is not None:
        model.load_adapter(CRITIQUE_MODEL_DIR, "critiquer", is_trainable=True)
        response_template = "\n\nCritique:"
        # skip '<s>' and '▁'
        response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )[1:]
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        )
        critique_training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=20,  # critical for memory usage
            learning_rate=CRITIQUER_LEARNING_RATE,
            logging_steps=1,
            num_train_epochs=1,
            report_to=["tensorboard"],
            eval_steps=None,
            # lr_scheduler_type="constant",
            # There are probably less steps actually, but this gets the lr to change more slowly
            max_steps=len(dataset) * RLOO_K * 2 // BATCH_SIZE,
        )
        # Needed for the optimizer to be initialized correctly
        model.set_adapter("critiquer")
        critique_trainer = IterativeSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=critique_training_args,
        )
    else:
        critique_trainer = None

    mini_batch_size = 28
    config = RLOOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=mini_batch_size,
        local_rollout_forward_batch_size=mini_batch_size,
        num_train_epochs=1,
        num_ppo_epochs=PPO_EPOCHS,
        gradient_accumulation_steps=BATCH_SIZE // mini_batch_size,
        kl_coef=KL_COEFF,
        rloo_k=RLOO_K,
        learning_rate=LEARNING_RATE,
        save_steps=16,
        logging_steps=1,
        report_to=["tensorboard"],
        temperature=1.0,
        stop_token="eos",
        # Technically doesn't fit the longest possible response
        response_length=42,
        num_sample_generations=0,
        # Should help the critiquer get a head start
        warmup_ratio=0.08,
        # lr_scheduler_type="constant",
    )
    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        model=model,
        policy_adapter="cluer",
        ref_policy_adapter="ref",
        reward_function=get_reward_function(
            overseer,
            tokenizer,
            model,
            critique_trainer,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)  # this should save all adapters


def get_reward_function(
    overseer: OverSeer,
    tokenizer: PreTrainedTokenizer,
    model: PeftModel,
    critique_trainer: IterativeSFTTrainer | None,
):
    def reward_function(postprocessed_query_responses: torch.Tensor) -> torch.Tensor:
        current_adapter = model.active_adapter
        query_responses = tokenizer.batch_decode(
            postprocessed_query_responses, skip_special_tokens=True
        )
        queries = []
        responses = []
        for query_response in query_responses:
            try:
                query, response = query_response.split("\n\n", maxsplit=1)
            except ValueError:
                logger.error(f"Error splitting query and response: {query_response}")
                raise
            queries.append(query)
            responses.append(response)

        games = [Game.parse(query) for query in queries]

        logger.debug("Parsing Clues")
        clues = [safe(Clue.parse_response, response) for response in responses]

        if CRITIQUE_MODEL_DIR is not None:
            logger.debug("Generating Critiques")
            model.set_adapter("critiquer")
            with torch.inference_mode():
                # TODO: don't hard code this 2
                critiques_per_clue = 2
                generation_batch_size = 32
                critique_prompts = [
                    f"{query_response.strip()}\n\nCritique:"
                    for (clue, query_response) in zip(clues, query_responses)
                    if clue is not None and clue.targets
                ]
                critique_texts_partitioned = []
                for batch_prompts in partition_all(
                    generation_batch_size, critique_prompts
                ):
                    batch_inputs = tokenizer(
                        batch_prompts, return_tensors="pt", padding=True
                    ).to(model.device)
                    # TODO: make this work if generating only 1 critique per clue
                    batch_output_tokens = model.generate(
                        **batch_inputs,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        max_new_tokens=18,
                        num_beams=critiques_per_clue,
                        num_beam_groups=critiques_per_clue,
                        diversity_penalty=1.0,
                        num_return_sequences=critiques_per_clue,
                        early_stopping=True,
                    )
                    batch_output_texts = tokenizer.batch_decode(
                        batch_output_tokens,
                        skip_special_tokens=True,
                    )
                    batch_texts = [
                        [
                            line
                            for line in output.splitlines()
                            if line.startswith("Critique:")
                        ][0]
                        for output in batch_output_texts
                    ]
                    critique_texts_partitioned += list(
                        partition_all(critiques_per_clue, batch_texts)
                    )

                iter_critique_texts_partitioned = iter(critique_texts_partitioned)

            logger.debug("Parsing Critiques")
            critiques = [
                [
                    critique
                    for output in next(iter_critique_texts_partitioned, None)  # type: ignore
                    if (critique := safe(Critique.parse_response, output)) is not None
                ]
                if clue is not None and clue.targets
                else []
                for clue in clues
            ]
            assert next(iter_critique_texts_partitioned, None) is None
            assert len(critiques) == len(clues) == len(games)
        else:
            critiques = [[] for _ in games]

        logger.debug("Evaluating and Overseeing")
        evaluations = [
            safe(evaluate_clue, game, ClueCritiques(clue=clue, critiques=critiques))
            if clue is not None
            else None
            for game, clue, critiques in zip(games, clues, critiques)
        ]
        oversights = [
            overseer.oversee(e) if e is not None else None for e in evaluations
        ]
        mean_true_score = sum(e.score for e in evaluations if e is not None) / (
            len([e for e in evaluations if e is not None]) or 1
        )
        logger.info(f"Mean true score: {mean_true_score:.3f}")

        calibrate_p = approximate_calibrate_p(oversights, games)
        logger.info(f"Calibrate p: {calibrate_p:.3f}")
        rewards = torch.tensor(
            [
                overseer.reward(g, o, KL_COEFF, calibrate_p)
                - ADVERSARIAL_ALPHA * o.ground_truth_score
                if o is not None
                # TODO: not sure what to put here, this is just to get it to learn the clue whitelist
                else -3.0
                for g, o in zip(games, oversights)
            ]
        )
        mean_reward = torch.mean(rewards)
        logger.info(f"Mean reward: {mean_reward:.3f}")
        reward_counts = Counter([r.item() for r in rewards])
        reward_counts_str = ", ".join(
            f"{k:.3f}: {v}" for k, v in sorted(reward_counts.most_common())
        )
        logger.info(f"Reward counts: {{{reward_counts_str}}}")

        if critique_trainer is not None:
            model.set_adapter("critiquer")  # just in case
            accepted_critiques_text = [
                f"{g}\n\n{o.clue_critiques.clue}\n\n{o.deciding_critique}\n"
                for g, o in zip(games, oversights)
                if o is not None and o.deciding_critique is not None
            ]
            rejected_critiques_text = [
                f"{g}\n\n{o.clue_critiques.clue}\n\n{c}\n"
                for g, o in zip(games, oversights)
                if o is not None
                for c in o.clue_critiques.critiques
                if o.deciding_critique is None
                or (o.comparisons_performed == 2 and c != o.deciding_critique)
            ]
            if accepted_critiques_text:
                logger.info(
                    f"Training critiquer on {len(accepted_critiques_text)} good critiques"
                )
                critique_trainer.step_accumulate(accepted_critiques_text, True)

            if rejected_critiques_text:
                logger.info(
                    f"Training critiquer on {len(rejected_critiques_text)} bad critiques"
                )
                critique_trainer.step_accumulate(
                    rejected_critiques_text,
                    False,
                    # Having a lot of weight on this causes the training to just diverge to nonsense
                    len(accepted_critiques_text) / len(rejected_critiques_text) * 0.05,
                )
                critique_trainer.step_update()
            else:
                logger.warning(
                    "All critiques good or all critiques bad, something is probably going wrong"
                )

        for g, o in zip(games, oversights):
            if o is not None:
                p_set = PreferenceSet(
                    game=g,
                    overseer=overseer,
                    oversights=[o],
                    adversarial_alpha=ADVERSARIAL_ALPHA,
                )
                print(p_set.model_dump_json())

        model.set_adapter(current_adapter)
        return rewards

    return reward_function


def load_game_dataset(dataset_file: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    data = [
        {
            "input_ids": tokenizer(
                f"{Game.model_validate_json(line)}\n\nClue:", padding=False
            )["input_ids"]
        }
        for line in Path(dataset_file).read_text().splitlines()
    ]
    dataset = Dataset.from_list(data)
    return dataset


def safe[T](f: Callable[..., T], *args) -> T | None:
    try:
        return f(*args)
    except Exception as e:
        str_inputs = [arg for arg in args if isinstance(arg, str)]
        if str_inputs:
            logging.error(
                f"Error running {f.__name__} with str inputs {str_inputs}: {e}"
            )
        else:
            logging.error(f"Error running {f.__name__}: {e}")
        return None


@app.command()
def robust():
    if CRITIQUE_MODEL_DIR:
        overseer = RobustJudgeOverSeer()
    else:
        overseer = RobustOverSeer()
    main(overseer)


@app.command()
def neglect_last_n(neglect_words: NonNegativeInt):
    if CRITIQUE_MODEL_DIR:
        raise ValueError("NeglectLastNOverSeer does not support critiques")
    overseer = NeglectLastNOverSeer(neglect_words=neglect_words)
    main(overseer)


@app.command()
def negligent_biased(
    neglect_words: NonNegativeInt,
    bias_neglected_words: NonNegativeInt,
    bias_non_neglected_words: NonNegativeInt,
    bias_factor: NonNegativeFloat,
):
    if CRITIQUE_MODEL_DIR:
        overseer = NegligentBiasedJudgeOverSeer(
            neglect_words=neglect_words,
            bias_neglected_words=bias_neglected_words,
            bias_non_neglected_words=bias_non_neglected_words,
            bias_factor=bias_factor,
        )
    else:
        overseer = NegligentBiasedOverSeer(
            neglect_words=neglect_words,
            bias_neglected_words=bias_neglected_words,
            bias_non_neglected_words=bias_non_neglected_words,
            bias_factor=bias_factor,
        )
    main(overseer)


@app.command()
def negligent_biased_base(
    neglect_words: NonNegativeInt,
    bias_neglected_words: NonNegativeInt,
    bias_non_neglected_words: NonNegativeInt,
    bias_factor: NonNegativeFloat,
):
    overseer = NegligentBiasedBaseOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
    )
    main(overseer)


if __name__ == "__main__":
    app()
