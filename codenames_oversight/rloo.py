import logging
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import Dataset
from peft import PeftModelForCausalLM  # type: ignore
from pydantic import NonNegativeFloat, NonNegativeInt
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from trl import set_seed
from trl.trainer.rloo_config import RLOOConfig

from .evaluate_clue import evaluate_clue
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


@app.callback()
def set_params(
    dataset_file: str,
    model_dir: str,
    output_dir: str,
    critique_model_dir: Optional[str] = None,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    batch_size: int = 448,
    cluer_learning_rate: float = 2e-5,
    cluer_kl_coeff: float = 0.06,
    cluer_ppo_epochs: int = 4,
    cluer_rloo_k: int = 4,
    critiquer_learning_rate: float = 4e-5,
    critiquer_kl_coeff: float = 0.2,
    critiquer_ppo_epochs: int = 4,
    critiquer_rloo_k: int = 2,
    adversarial_alpha: float = 0.0,
):
    global DATASET_FILE
    global MODEL_DIR
    global OUTPUT_DIR
    global CRITIQUE_MODEL_DIR
    global BASE_MODEL
    global BATCH_SIZE
    global CLUER_LEARNING_RATE
    global CLUER_KL_COEFF
    global CLUER_PPO_EPOCHS
    global CLUER_RLOO_K
    global CRITIQUER_LEARNING_RATE
    global CRITIQUER_KL_COEFF
    global CRITIQUER_PPO_EPOCHS
    global CRITIQUER_RLOO_K
    global ADVERSARIAL_ALPHA
    DATASET_FILE: str = dataset_file
    MODEL_DIR: str = model_dir
    OUTPUT_DIR: str = output_dir
    CRITIQUE_MODEL_DIR: Optional[str] = critique_model_dir
    BASE_MODEL: str = base_model
    BATCH_SIZE: int = batch_size
    CLUER_LEARNING_RATE: float = cluer_learning_rate
    CLUER_KL_COEFF: float = cluer_kl_coeff
    CLUER_PPO_EPOCHS: int = cluer_ppo_epochs
    CLUER_RLOO_K: int = cluer_rloo_k
    CRITIQUER_LEARNING_RATE: float = critiquer_learning_rate
    CRITIQUER_KL_COEFF: float = critiquer_kl_coeff
    CRITIQUER_PPO_EPOCHS: int = critiquer_ppo_epochs
    CRITIQUER_RLOO_K: int = critiquer_rloo_k
    ADVERSARIAL_ALPHA: float = adversarial_alpha


def main(overseer: OverSeer):
    set_seed(0)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_eos_token=False, padding_side="left")  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_game_dataset(DATASET_FILE, tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModelForCausalLM.from_pretrained(base_model, MODEL_DIR, is_trainable=False, adapter_name="cluer_ref")
    model.load_adapter(MODEL_DIR, "cluer", is_trainable=True)

    if CRITIQUE_MODEL_DIR is not None:
        model.load_adapter(CRITIQUE_MODEL_DIR, "critiquer_ref", is_trainable=False)
        model.load_adapter(CRITIQUE_MODEL_DIR, "critiquer", is_trainable=True)
        critiquer_mini_batch_size = 20
        critiquer_config = RLOOConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=critiquer_mini_batch_size,
            local_rollout_forward_batch_size=critiquer_mini_batch_size,
            num_train_epochs=1,
            num_ppo_epochs=CRITIQUER_PPO_EPOCHS,
            gradient_accumulation_steps=BATCH_SIZE // critiquer_mini_batch_size,
            kl_coef=CRITIQUER_KL_COEFF,
            rloo_k=CRITIQUER_RLOO_K,
            learning_rate=CRITIQUER_LEARNING_RATE,
            save_steps=16,
            logging_steps=1,
            report_to=["tensorboard"],
            temperature=1.0,
            stop_token="eos",
            response_length=18,
            num_sample_generations=0,
            # lr_scheduler_type="constant",
        )
        critique_trainer = RLOOTrainer(
            config=critiquer_config,
            tokenizer=tokenizer,
            model=model,
            policy_adapter="critiquer",
            ref_policy_adapter="critiquer_ref",
        )
    else:
        critique_trainer = None

    mini_batch_size = 28
    config = RLOOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=mini_batch_size,
        local_rollout_forward_batch_size=mini_batch_size,
        num_train_epochs=1,
        num_ppo_epochs=CLUER_PPO_EPOCHS,
        gradient_accumulation_steps=BATCH_SIZE // mini_batch_size,
        kl_coef=CLUER_KL_COEFF,
        rloo_k=CLUER_RLOO_K,
        learning_rate=CLUER_LEARNING_RATE,
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
    cluer_trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        model=model,
        policy_adapter="cluer",
        ref_policy_adapter="cluer_ref",
        train_dataset=dataset,
    )
    dataloader = cluer_trainer.get_train_dataloader()
    for data in tqdm(dataloader, total=config.num_updates, desc="Running RLOO"):
        queries = data["input_ids"].to(cluer_trainer.accelerator.device)
        postprocessed_query_response_tokens = cluer_trainer.rollout(queries)
        query_responses = tokenizer.batch_decode(postprocessed_query_response_tokens, skip_special_tokens=True)
        queries_text = []
        responses_text = []
        for query_response in query_responses:
            try:
                query, response = query_response.split("\n\n", maxsplit=1)
            except ValueError:
                logger.error(f"Error splitting query and response: {query_response}")
                raise
            queries_text.append(query)
            responses_text.append(response)

        games = [Game.parse(query) for query in queries_text]

        logger.debug("Parsing Clues")
        clues = [safe(Clue.parse_response, response) for response in responses_text]
        evaluations = [
            [safe(evaluate_clue, game, ClueCritiques(clue=clue, critiques=[])) if clue is not None else None]
            for game, clue in zip(games, clues)
        ]
        if critique_trainer is not None:
            for es in evaluations:
                for _ in range(critique_trainer.args.rloo_k - 1):
                    es.append(deepcopy(es[0]))
            logger.debug("Generating Critiques")
            critique_prompts = [
                f"{query_response.strip()}\n\nCritique:"
                for (clue, query_response) in zip(clues, query_responses)
                if clue is not None and clue.targets
            ]
            queries: torch.Tensor = tokenizer(critique_prompts, return_tensors="pt", padding=True)["input_ids"]  # type: ignore
            postprocessed_query_response_tokens = critique_trainer.rollout(
                queries.to(critique_trainer.accelerator.device)
            )
            query_responses = tokenizer.batch_decode(postprocessed_query_response_tokens, skip_special_tokens=True)
            critique_texts = []
            for query_response in query_responses:
                try:
                    _, _, response = query_response.split("\n\n", maxsplit=2)
                except ValueError:
                    logger.error(f"Error splitting critiques response: {query_response}")
                    raise
                critique_texts.append(response)

            critiques = [safe(Critique.parse_response, response) for response in critique_texts]
            iter_critiques = iter(critiques)
            for i in range(critique_trainer.args.rloo_k):
                for es in evaluations:
                    e = es[i]
                    if e is None:
                        continue
                    try:
                        critique = next(iter_critiques)
                    except StopIteration:
                        raise ValueError("Not enough critiques generated")
                    if critique is not None:
                        e.clue_critiques.critiques = [critique]

            try:
                next(iter_critiques)
                raise ValueError("Too many critiques generated")
            except StopIteration:
                pass

        oversights = [[overseer.oversee(e) if e is not None else None for e in es] for es in evaluations]
        mean_true_score = sum(es[0].score for es in evaluations if es[0] is not None) / (
            len([es for es in evaluations if es[0] is not None]) or 1
        )
        logger.info(f"Mean true score: {mean_true_score:.3f}")

        calibrate_p = approximate_calibrate_p([o for os in oversights for o in os], games)
        logger.info(f"Calibrate p: {calibrate_p:.3f}")
        rewards = torch.tensor(
            [
                min(
                    [
                        overseer.reward(g, o, CLUER_KL_COEFF, calibrate_p) - ADVERSARIAL_ALPHA * o.ground_truth_score
                        if o is not None
                        # TODO: not sure what to put here, this is just to get it to learn the clue whitelist
                        else -3.0
                        for o in os
                    ]
                )
                for g, os in zip(games, oversights)
            ]
        )
        mean_reward = torch.mean(rewards)
        logger.info(f"Mean reward: {mean_reward:.3f}")
        reward_counts = Counter([r.item() for r in rewards])
        reward_counts_str = ", ".join(f"{k:.3f}: {v}" for k, v in sorted(reward_counts.most_common()))
        logger.info(f"Reward counts: {{{reward_counts_str}}}")

        if critique_trainer is not None:
            critique_rewards = []
            for i in range(critique_trainer.args.rloo_k):
                for os in oversights:
                    o = os[i]
                    if o is None:
                        continue
                    if o.deciding_critique is not None:
                        reward = 1.0
                    elif o.clue_critiques.critiques:
                        reward = 0.0
                    else:
                        reward = -1.0
                    critique_rewards.append(reward)
            assert len(critique_rewards) == len(critiques)  # type: ignore
            critique_rewards = torch.tensor(critique_rewards)
            critique_mean_reward = torch.mean(critique_rewards)
            logger.info(f"Critique mean reward: {critique_mean_reward:.3f}")
            critique_reward_counts = Counter([r.item() for r in critique_rewards])
            critique_reward_counts_str = ", ".join(
                f"{k:.3f}: {v}" for k, v in sorted(critique_reward_counts.most_common())
            )
            logger.info(f"Critique reward counts: {{{critique_reward_counts_str}}}")
            critique_trainer.rl_step(critique_rewards)

        cluer_trainer.rl_step(rewards)
        for g, os in zip(games, oversights):
            for o in os:
                if o is not None:
                    p_set = PreferenceSet(
                        game=g,
                        overseer=overseer,
                        oversights=[o],
                        adversarial_alpha=ADVERSARIAL_ALPHA,
                    )
                    print(p_set.model_dump_json())

    cluer_trainer.end_train()
    if critique_trainer is not None:
        critique_trainer.end_train()
    model.save_pretrained(OUTPUT_DIR)  # this should save all adapters


def load_game_dataset(dataset_file: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    data = [
        {"input_ids": tokenizer(f"{Game.model_validate_json(line)}\n\nClue:", padding=False)["input_ids"]}
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
            logging.error(f"Error running {f.__name__} with str inputs {str_inputs}: {e}")
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
