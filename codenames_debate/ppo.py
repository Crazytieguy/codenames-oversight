import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed

from .evaluate_clue import evaluate_clue
from .models import Clue, ClueCritiques, Game

logging.basicConfig(level=logging.INFO)
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    dataset_file: str,
    model_dir: str,
    output_dir: str,
    reference_model: Optional[str] = None,
    learning_rate: float = 5e-6,
    batch_size: int = 64,
):
    set_seed(0)
    dataset = load_game_dataset(dataset_file)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    )
    if reference_model is None:
        reference_model = model_dir
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        reference_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        is_trainable=False,
    )
    mini_batch_size = 8  # important for memory usage
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        ppo_epochs=1,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=batch_size // mini_batch_size,
        init_kl_coef=0.05,
        log_with="tensorboard",
        project_kwargs={"logging_dir": f"{output_dir}/logs"},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, add_eos_token=False, padding_side="left"
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
                max_new_tokens=128,
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
            rewards = [
                torch.tensor(float(e.score if e is not None else -10))
                for e in evaluations
            ]
            for e in evaluations:
                if e is not None:
                    print(e.model_dump_json())

        with timer("ppo step"):
            stats = ppo_trainer.step(inputs, outputs, rewards)  # type: ignore
            ppo_trainer.log_stats(stats, batch, rewards)  # type: ignore

    ppo_trainer.save_pretrained(output_dir)


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
