import sys
from typing import Optional

import torch
import typer
from outlines.generate import regex  # type: ignore
from outlines.models.transformers import Transformers
from outlines.samplers import multinomial
from peft import AutoPeftModelForCausalLM  # type: ignore
from toolz.itertoolz import partition_all
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .models import (
    CLUE_WORDS,
    GAME_WORDS,
    Clue,
    ClueCritiques,
    Game,
    InferenceSample,
    generate_game,
)

app = typer.Typer(pretty_exceptions_show_locals=False)

GAME_WORD = f"(?:{"|".join(GAME_WORDS)})"
CLUE_WORD = f"(?:{"|".join(CLUE_WORDS)})"
CLUE_PATTERN = rf"Clue: {CLUE_WORD}\n"
TARGETS_PATTERN = rf"Targets: {GAME_WORD}(?:, {GAME_WORD})*\n"
FULL_PATTERN = rf"{CLUE_PATTERN}{TARGETS_PATTERN}\n"


@app.command()
def main(
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    random_games: Optional[int] = None,
    clues_per_game: int = 2,
    batch_size: int = 16,
    temperature: Optional[float] = None,
):
    "Give some clues"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_attentions=True,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_attentions=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, add_eos_token=False, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = Transformers(model, tokenizer)  # type: ignore
    sampler = multinomial(clues_per_game, temperature=temperature)
    generator = regex(model, FULL_PATTERN, sampler)
    rng = torch.Generator(device="cuda")
    rng.manual_seed(42)

    if random_games is not None:
        games = [generate_game() for _ in range(random_games)]
    else:
        games = [Game.model_validate_json(line) for line in sys.stdin]

    for batch in tqdm(
        partition_all(batch_size, games),
        desc="Generating clues",
        total=len(games) // batch_size,
    ):
        prompts = [f"{game}\n\n" for game in batch]
        outputs = generator(prompts, max_tokens=64, stop_at="\n\n", rng=rng)
        if clues_per_game == 1:
            outputs = [[output] for output in outputs]  # type: ignore
        for game, outputs in zip(batch, outputs):  # type: ignore
            clues = [Clue.parse_response(output) for output in outputs]
            sample = InferenceSample(
                game=game, clue_critiques=[ClueCritiques(clue=clue) for clue in clues]
            )
            print(sample.model_dump_json())


if __name__ == "__main__":
    app()
