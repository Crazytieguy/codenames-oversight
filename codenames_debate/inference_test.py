import json
from pathlib import Path

import torch
import typer
from peft import AutoPeftModelForCausalLM  # type: ignore
from toolz.itertoolz import partition_all
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .models import generate_game


def main(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    model_dir: str = "./models/llama-7b-clue-giving",
    output_file: Path = Path("data/inference-llama-7b-clue-giving.jsonl"),
    num_games: int = 16,
    two_per_game: bool = True,  # for DPO
    batch_size: int = 4,
):
    "Give some clues and evaluate them"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    games = [generate_game() for _ in range(num_games)]
    out = []
    for batch in partition_all(batch_size, games):
        prompts = [f"{game}\n\nClue:" for game in batch]
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(
            model.device
        )
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            num_beams=2 if two_per_game else 1,
            num_return_sequences=2 if two_per_game else 1,
        )
        output_texts = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )
        out.append(json.dumps(output_texts))
    
    output_file.write_text("".join(line + "\n" for line in out))


if __name__ == "__main__":
    typer.run(main)
