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
    GAME_WORDS,
    Clue,
    ClueCritiques,
    Game,
    InferenceSample,
    generate_game,
)

app = typer.Typer(pretty_exceptions_show_locals=False)

CAPITAL_GAME_WORD = f"(?:{"|".join(GAME_WORDS)})"
TITLE_GAME_WORD = f"(?:{"|".join(w.title() for w in GAME_WORDS)})"
CLUE_PATTERN = rf"Clue: (?!{TITLE_GAME_WORD})[A-Z][a-z]*\n"
TARGETS_PATTERN = rf"Targets: {CAPITAL_GAME_WORD}(?:, {CAPITAL_GAME_WORD})*\n"


@app.command()
def main(
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    random_games: Optional[int] = None,
    clues_per_game: int = 2,
    batch_size: int = 8,
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
    sampler = multinomial(clues_per_game)
    clue_generator = regex(model, CLUE_PATTERN, sampler)
    targets_generator = regex(model, TARGETS_PATTERN)
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
        clue_prompts = [f"{game}\n\n" for game in batch]
        clue_outputs = clue_generator(
            clue_prompts, max_tokens=10, stop_at="\n", rng=rng
        )
        if clues_per_game == 1:
            clue_outputs = [[output] for output in clue_outputs]  # type: ignore
        clue_outputs = [
            [output.strip() + "\n" for output in outputs] for outputs in clue_outputs
        ]
        target_prompts = [
            f"{clue_prompt}{output}"
            for clue_prompt, outputs in zip(clue_prompts, clue_outputs)
            for output in outputs
        ]
        target_outputs = targets_generator(
            target_prompts, max_tokens=32, stop_at="\n", rng=rng
        )
        for game, clue_outputs, target_outputs in zip(
            batch, clue_outputs, partition_all(clues_per_game, target_outputs)
        ):  # type: ignore
            clues = [
                Clue.parse_response(clue_output + target_output)
                for clue_output, target_output in zip(clue_outputs, target_outputs)
            ]
            sample = InferenceSample(
                game=game, clue_critiques=[ClueCritiques(clue=clue) for clue in clues]
            )
            print(sample.model_dump_json())


if __name__ == "__main__":
    app()
