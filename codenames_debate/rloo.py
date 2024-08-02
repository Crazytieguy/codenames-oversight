import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch
import typer
from datasets import Dataset
from peft import LoraConfig, get_peft_model  # type: ignore
from pydantic import NonNegativeFloat, NonNegativeInt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from trl import set_seed
from trl.trainer.rloo_config import RLOOConfig

from .evaluate_clue import evaluate_clue
from .models import Clue, ClueCritiques, Game
from .oversight import (
    NeglectLastNOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
    RobustJudgeOverSeer,
    RobustOverSeer,
)
from .ppo_reward import (
    approximate_calibrate_p,
    reward_accept,
    reward_reject,
)
from .rloo_trainer import RLOOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


DATASET_FILE: str
MODEL_DIR: str
OUTPUT_DIR: str
LEARNING_RATE: float
BATCH_SIZE: int
KL_COEFF: float
PPO_EPOCHS: int
ADVERSARIAL_ALPHA: float


@app.callback()
def set_params(
    dataset_file: str,
    model_dir: str,
    output_dir: str,
    learning_rate: float = 5e-5,
    batch_size: int = 128,
    kl_coeff: float = 0.05,
    ppo_epochs: int = 4,
    adversarial_alpha: float = 0.0,
):
    global DATASET_FILE
    global MODEL_DIR
    global OUTPUT_DIR
    global LEARNING_RATE
    global BATCH_SIZE
    global KL_COEFF
    global PPO_EPOCHS
    global ADVERSARIAL_ALPHA
    DATASET_FILE = dataset_file
    MODEL_DIR = model_dir
    OUTPUT_DIR = output_dir
    LEARNING_RATE = learning_rate
    BATCH_SIZE = batch_size
    KL_COEFF = kl_coeff
    PPO_EPOCHS = ppo_epochs
    ADVERSARIAL_ALPHA = adversarial_alpha


def main(overseer: OverSeer):
    set_seed(0)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        r=256,
        lora_alpha=128,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(ref_model, peft_config)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, add_eos_token=False, padding_side="left"
    )  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_game_dataset(DATASET_FILE, tokenizer)
    # critical for memory usage
    mini_batch_size = 16
    config = RLOOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=mini_batch_size,
        num_train_epochs=1,
        num_ppo_epochs=PPO_EPOCHS,
        gradient_accumulation_steps=BATCH_SIZE // mini_batch_size,
        kl_coef=KL_COEFF,
        rloo_k=4,
        learning_rate=LEARNING_RATE,
        save_steps=16,
        logging_steps=1,
        temperature=0.7,  # TODO: this is the default value and I don't know why it's not 1.0.
    )
    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=model,
        ref_policy=ref_model,
        reward_function=get_reward_function(overseer, tokenizer),
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)


def get_reward_function(overseer: OverSeer, tokenizer: PreTrainedTokenizer):
    def reward_function(postprocessed_query_responses: torch.Tensor) -> torch.Tensor:
        query_responses = tokenizer.batch_decode(
            postprocessed_query_responses, skip_special_tokens=True
        )
        queries = []
        responses = []
        for query_response in query_responses:
            query, response = query_response.split("\n\n")
            queries.append(query)
            responses.append(response)

        games = [Game.parse(query) for query in queries]
        clues = [safe(Clue.parse_response, response) for response in responses]
        evaluations = [
            safe(evaluate_clue, game, ClueCritiques(clue=clue))
            if clue is not None
            else None
            for game, clue in zip(games, clues)
        ]
        mean_true_score = sum(e.score for e in evaluations if e is not None) / (
            len([e for e in evaluations if e is not None]) or 1
        )
        logger.info(f"Mean true score: {mean_true_score}")
        oversights = [
            overseer.oversee(e) if e is not None else None for e in evaluations
        ]
        calibrate_p = approximate_calibrate_p(oversights, games)
        logger.info(f"Calibrate p: {calibrate_p}")
        rewards = [
            torch.tensor(
                (
                    reward_reject(
                        bad_words_in_game=len(g.bad_words),
                        n_targets=len(o.valid_targets),
                        kl_coeff=KL_COEFF,
                        calibrated_p=calibrate_p,
                    )
                    if o.deciding_critique is not None
                    else reward_accept(
                        bad_words_in_game=len(g.bad_words),
                        n_targets=len(o.valid_targets),
                        calibrated_p=calibrate_p,
                        kl_coeff=KL_COEFF,
                    )
                )
                - ADVERSARIAL_ALPHA * o.ground_truth_score
                if o is not None
                # TODO: not sure what to put here, this is just to get it to learn the clue whitelist
                else -1.0
            )
            for g, o in zip(games, oversights)
        ]
        reward_counts = Counter([r.item() for r in rewards])
        logger.info(f"Reward counts: {dict(sorted(reward_counts.most_common()))}")
        for g, o in zip(games, oversights):
            if o is not None:
                p_set = PreferenceSet(game=g, overseer=overseer, oversights=[o])
                print(p_set.model_dump_json())

        return torch.tensor(rewards)

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
        logging.error(f"Error running {f.__name__}: {e}")
        return None


@app.command()
def robust():
    overseer = RobustOverSeer()
    main(overseer)


@app.command()
def robust_judge():
    overseer = RobustJudgeOverSeer()
    main(overseer)


@app.command()
def neglect_last_n(neglect_words: NonNegativeInt):
    overseer = NeglectLastNOverSeer(neglect_words=neglect_words)
    main(overseer)


@app.command()
def negligent_biased(
    neglect_words: NonNegativeInt,
    bias_neglected_words: NonNegativeInt,
    bias_non_neglected_words: NonNegativeInt,
    bias_factor: NonNegativeFloat,
):
    overseer = NegligentBiasedOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
    )
    main(overseer)


if __name__ == "__main__":
    app()
