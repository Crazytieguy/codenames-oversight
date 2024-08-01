import logging
import time
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import torch
import typer
from datasets import Dataset
from peft import LoraConfig, get_peft_model  # type: ignore
from pydantic import NonNegativeFloat, NonNegativeInt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    set_seed,
)

from codenames_debate.oversight import (
    NeglectLastNOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
    RobustJudgeOverSeer,
    RobustOverSeer,
)

from .evaluate_clue import evaluate_clue
from .models import Clue, ClueCritiques, Game
from .ppo_reward import approximate_calibrate_p, reward_accept, reward_reject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


DATASET_FILE: str
MODEL_DIR: str
OUTPUT_DIR: str
BASE_MODEL: str
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
    base_model: str = "meta-llama/Llama-2-7b-hf",
    learning_rate: float = 4e-5,
    batch_size: int = 256,
    kl_coeff: float = 0.05,
    ppo_epochs: int = 4,
    adversarial_alpha: float = 0.0,
):
    global DATASET_FILE
    global MODEL_DIR
    global OUTPUT_DIR
    global BASE_MODEL
    global LEARNING_RATE
    global BATCH_SIZE
    global KL_COEFF
    global PPO_EPOCHS
    global ADVERSARIAL_ALPHA
    DATASET_FILE = dataset_file
    MODEL_DIR = model_dir
    OUTPUT_DIR = output_dir
    BASE_MODEL = base_model
    LEARNING_RATE = learning_rate
    BATCH_SIZE = batch_size
    KL_COEFF = kl_coeff
    PPO_EPOCHS = ppo_epochs
    ADVERSARIAL_ALPHA = adversarial_alpha


def main(overseer: OverSeer):
    set_seed(0)
    dataset = load_game_dataset(DATASET_FILE)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(ref_model, peft_config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        peft_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    )

    mini_batch_size = 8  # important for memory usage

    ppo_config = PPOConfig(
        learning_rate=LEARNING_RATE,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=BATCH_SIZE // mini_batch_size,
        init_kl_coef=KL_COEFF,
        adap_kl_ctrl=False,  # reward function relies on fixed KL
        log_with="tensorboard",
        project_kwargs={"logging_dir": f"{OUTPUT_DIR}/logs"},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, add_eos_token=False, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer: PPOTrainer = PPOTrainer(
        ppo_config,
        model=model,
        ref_model=ref_model,
        dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )  # type: ignore

    for batch in tqdm(
        ppo_trainer.dataloader,
        total=len(dataset) / ppo_config.batch_size,
        desc="Training",
    ):
        with timer("generation"):
            inputs = [torch.tensor(tokenizer.encode(q)) for q in batch["query"]]  # type: ignore
            outputs = ppo_trainer.generate(
                inputs,
                return_prompt=False,
                # important for preventing KL term from dominating weirdly
                # see https://huggingface.co/docs/trl/en/how_to_train#how-to-generate-text-for-training
                min_length=-1,
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=64,
            )
            output_texts = [tokenizer.decode(o.squeeze()) for o in outputs]  # type: ignore

        with timer("evaluation"):
            games = [
                Game.parse(query.removesuffix("Clue:"))
                for query in batch["query"]  # type: ignore
            ]
            clues = [
                safe(Clue.parse_response, f"Clue: {response}")
                for response in output_texts
            ]
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

        with timer("ppo step"):
            stats = ppo_trainer.step(inputs, outputs, rewards)  # type: ignore
            stats.update(
                {"env/mean_true_score": mean_true_score, "env/calibrate_p": calibrate_p}
            )
            ppo_trainer.log_stats(stats, batch, rewards)  # type: ignore

    ppo_trainer.save_pretrained(OUTPUT_DIR)


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


def safe[T](f: Callable[..., T], *args) -> T | None:
    try:
        return f(*args)
    except Exception as e:
        logging.error(f"Error running {f.__name__}: {e}")
        return None


def collator(samples: list[dict]) -> dict:
    return {
        "query": [s["query"] for s in samples],
    }


def load_game_dataset(dataset_file: str) -> Dataset:
    data = [
        {"query": f"{Game.model_validate_json(line)}\n\nClue:"}
        for line in Path(dataset_file).read_text().splitlines()
    ]
    dataset = Dataset.from_list(data)
    return dataset


@contextmanager
def timer(name: str):
    start = time.time()
    yield
    logging.info(f"{name} took {time.time() - start:.2f}s")


if __name__ == "__main__":
    app()
