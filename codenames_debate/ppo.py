import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
import typer
from accelerate import Accelerator
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    is_xpu_available,
    set_seed,
)

from .evaluate_clue import evaluate_clue, parse_clue
from .models import Evaluation, Game

logging.basicConfig(level=logging.INFO)


def main(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    model_dir: str = "llama-7b-clue-giving-ppo",
    dataset_path: Path = Path("codenames_debate/ppo_dataset.jsonl"),
    ppo_evaluations_log: Path = Path("codenames_debate/ppo_evaluations.jsonl"),
    evaluation_concurrency: int = 8,
    fresh_start: bool = False,
):
    "Train a fine tuned LLM with PPO to generate better CodeNames clues."
    set_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    def prepare_sample(game: Game) -> dict:
        query = f"{game}\n\nClue:"
        return {
            "query": query,
            "input_ids": tokenizer.encode(query),
        }

    games = [
        Game.model_validate_json(line) for line in dataset_path.read_text().splitlines()
    ]

    resuming = not fresh_start and ppo_evaluations_log.exists()

    if resuming:
        evaluations = [
            Evaluation.model_validate_json(line)
            for line in ppo_evaluations_log.read_text().splitlines()
        ]
        game_strs_trained = {str(e.game) for e in evaluations}
        games = [game for game in games if str(game) not in game_strs_trained]
        logging.info(f"Resuming training, skipping {len(game_strs_trained)} games")

    samples = [prepare_sample(game) for game in games]

    # the games can't be inside the dataset, so keeping them separately
    game_by_query = {sample["query"]: game for game, sample in zip(games, samples)}
    dataset = Dataset.from_list(samples)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        f"./{model_dir}",
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    ppo_config = PPOConfig(
        learning_rate=1e-4,
        ppo_epochs=1,
        batch_size=16,
    )

    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 10,
    }

    try:
        for batch in tqdm(
            ppo_trainer.dataloader,  # type: ignore
            total=len(dataset) / ppo_config.batch_size,
            desc="Training",
        ):
            with timer("generation"):
                response_tensors = ppo_trainer.generate(
                    batch["input_ids"], return_prompt=False, **generation_kwargs
                )
                responses = [tokenizer.decode(r.squeeze()) for r in response_tensors]  # type: ignore

            with timer("evaluation"):
                games = [game_by_query[query] for query in batch["query"]]  # type: ignore
                clues = list(map(parse_clue, responses))
                with ThreadPoolExecutor(max_workers=evaluation_concurrency) as ex:
                    evaluations = list(ex.map(evaluate_clue, games, clues))

                rewards = [torch.tensor(e.reward) for e in evaluations]

            with timer("ppo step"):
                ppo_trainer.step(
                    batch["input_ids"],
                    response_tensors,  # type: ignore
                    rewards,  # type: ignore
                )

            logging.info(f"rewards={[e.reward for e in evaluations]}")

            with ppo_evaluations_log.open("a") as f:
                for evaluation in evaluations:
                    f.write(evaluation.model_dump_json() + "\n")

    finally:
        model.save_pretrained(model_dir)


def collator(samples: list[dict]) -> dict:
    return {
        "query": [s["query"] for s in samples],
        "input_ids": [torch.tensor(s["input_ids"]) for s in samples],
    }


@contextmanager
def timer(name: str):
    start = time.time()
    yield
    logging.info(f"{name} took {time.time() - start:.2f}s")


if __name__ == "__main__":
    typer.run(main)
