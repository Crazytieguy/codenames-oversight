import logging
import time
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
    RobustJudgeOverSeer,
    RobustOverSeer,
)

from .evaluate_clue import evaluate_clue
from .models import Clue, ClueCritiques, Game

logging.basicConfig(level=logging.INFO)
app = typer.Typer(pretty_exceptions_show_locals=False)


DATASET_FILE: str
MODEL_DIR: str
OUTPUT_DIR: str
BASE_MODEL: str
LEARNING_RATE: float
BATCH_SIZE: int


@app.callback()
def set_params(
    dataset_file: str,
    model_dir: str,
    output_dir: str,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    learning_rate: float = 1e-4,
    batch_size: int = 64,
):
    global DATASET_FILE
    global MODEL_DIR
    global OUTPUT_DIR
    global BASE_MODEL
    global LEARNING_RATE
    global BATCH_SIZE
    DATASET_FILE = dataset_file
    MODEL_DIR = model_dir
    OUTPUT_DIR = output_dir
    BASE_MODEL = base_model
    LEARNING_RATE = learning_rate
    BATCH_SIZE = batch_size


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
        ppo_epochs=1,
        batch_size=BATCH_SIZE,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=BATCH_SIZE // mini_batch_size,
        init_kl_coef=0.05,
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
                safe(Game.parse, query.removesuffix("Clue:"))
                for query in batch["query"]  # type: ignore
            ]
            clues = [
                safe(Clue.parse_response, f"Clue: {response}")
                for response in output_texts
            ]
            evaluations = [
                safe(evaluate_clue, game, ClueCritiques(clue=clue))
                if (game is not None and clue is not None)
                else None
                for game, clue in zip(games, clues)
            ]
            oversights = [
                overseer.oversee(e) if e is not None else None for e in evaluations
            ]
            rewards = [
                torch.tensor(float(o.expected_score if o is not None else -10))
                for o in oversights
            ]
            for e in evaluations:
                if e is not None:
                    print(e.model_dump_json())

        with timer("ppo step"):
            stats = ppo_trainer.step(inputs, outputs, rewards)  # type: ignore
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
    except Exception:
        logging.error(f"Error running {f.__name__} on inputs {args}")
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
