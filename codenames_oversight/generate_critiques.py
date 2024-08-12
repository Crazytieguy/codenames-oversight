import sys
from itertools import repeat

import torch
import typer
from outlines.generate import regex, text  # type: ignore
from outlines.models.transformers import Transformers
from peft import AutoPeftModelForCausalLM  # type: ignore
from toolz.itertoolz import partition_all, zip_longest
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .models import GAME_WORDS, Clue, Critique, Game, InferenceSample
from .oversight import PreferenceSet

app = typer.Typer(pretty_exceptions_show_locals=False)

GAME_WORD = f"(?:{'|'.join(GAME_WORDS)})"
GENERATION_PATTERN = rf"Critique: {GAME_WORD} > {GAME_WORD}\n"


@app.command()
def main(
    model_name_or_path: str,
    critiques_per_clue: int = 2,
    batch_size: int = 64,
):
    "Give some critiques"
    preference_sets = [PreferenceSet.model_validate_json(line) for line in sys.stdin]

    inference_samples = [
        InferenceSample(
            game=p.game, clue_critiques=[o.clue_critiques for o in p.oversights]
        )
        for p in preference_sets
        if all(o.clue_critiques.clue.targets for o in p.oversights)
    ]
    clues_per_game = len(inference_samples[0].clue_critiques)
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
    # generator = regex(model, GENERATION_PATTERN, sampler)
    generator = text(model)

    for batch in tqdm(
        partition_all(batch_size // clues_per_game, inference_samples),
        desc="Generating critiques",
        total=len(inference_samples) // (batch_size // clues_per_game),
    ):
        batch: list[InferenceSample]
        prompts = [
            prompt
            for sample in batch
            for clue_critique in sample.clue_critiques
            for prompt in repeat(
                format_prompt(sample.game, clue_critique.clue), critiques_per_clue
            )
        ]
        outputs = generator(prompts, max_tokens=24, stop_at="\n")
        outputs = partition_all(critiques_per_clue, outputs)
        for sample, game_outputs in zip_longest(
            batch, partition_all(clues_per_game, outputs)
        ):
            for clue_critiques, clue_outputs in zip_longest(
                sample.clue_critiques, game_outputs
            ):
                clue_critiques.critiques = list(
                    map(Critique.parse_response, clue_outputs)
                )
            print(sample.model_dump_json())


def format_prompt(game: Game, clue: Clue) -> str:
    return f"""\
{game}

{clue}

Critique:"""


if __name__ == "__main__":
    app()
