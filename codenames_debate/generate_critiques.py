import sys

import torch
import typer
from outlines.generate import regex  # type: ignore
from outlines.models.transformers import Transformers
from outlines.samplers import multinomial
from peft import AutoPeftModelForCausalLM  # type: ignore
from toolz.itertoolz import partition_all, zip_longest
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .models import GAME_WORDS, Clue, Critique, Game, InferenceSample

app = typer.Typer(pretty_exceptions_show_locals=False)

GAME_WORD = f"(?:{'|'.join(GAME_WORDS)})"
GENERATION_PATTERN = rf"Critique: {GAME_WORD} > {GAME_WORD}\n"


@app.command()
def main(
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    critiques_per_clue: int = 1,
    batch_size: int = 12,
):
    "Give some critiques"
    inference_samples = [
        InferenceSample.model_validate_json(line) for line in sys.stdin
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
    sampler = multinomial(critiques_per_clue)
    generator = regex(model, GENERATION_PATTERN, sampler)

    for batch in tqdm(
        partition_all(batch_size // clues_per_game, inference_samples),
        desc="Generating critiques",
        total=len(inference_samples) // (batch_size // clues_per_game),
    ):
        batch: list[InferenceSample]
        prompts = [
            format_prompt(sample.game, clue_critique.clue)
            for sample in batch
            for clue_critique in sample.clue_critiques
        ]
        outputs = generator(prompts, max_tokens=32, stop_at="\n")
        if critiques_per_clue == 1:
            outputs = [[output] for output in outputs]  # type: ignore
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

"""


if __name__ == "__main__":
    app()
