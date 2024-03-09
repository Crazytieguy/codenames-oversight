from pathlib import Path

import torch
import typer
from peft import AutoPeftModelForCausalLM  # type: ignore
from toolz.itertoolz import partition_all
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .models import (
    Clue,
    Critique,
    Game,
    InferenceSample,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    model_dir: str,
    clues_file: Path,
    critiques_per_clue: int = 1,
    batch_size: int = 12,
    diversity_penalty: float = 1.0,
):
    "Give some critiques"
    inference_samples = [
        InferenceSample.model_validate_json(line)
        for line in clues_file.read_text().splitlines()
    ]
    clues_per_game = len(inference_samples[0].clue_critiques)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, add_eos_token=False, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

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
            if isinstance(clue_critique.clue, Clue)
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        if critiques_per_clue > 1:
            outputs = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=32,
                num_beams=critiques_per_clue,
                num_beam_groups=critiques_per_clue,
                diversity_penalty=diversity_penalty,
                num_return_sequences=critiques_per_clue,
            )
        else:
            outputs = model.generate(**inputs, max_new_tokens=32)
        output_texts = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )
        read_outputs = 0
        for sample in batch:
            for clue_critique in sample.clue_critiques:
                if isinstance(clue_critique.clue, Clue):
                    clue_outputs = output_texts[
                        read_outputs : read_outputs + critiques_per_clue
                    ]
                    read_outputs += critiques_per_clue
                    clue_critique.critiques = [
                        Critique.parse_response(output.split("\n\n")[-1])
                        for output in clue_outputs
                    ]
            print(sample.model_dump_json())
        assert read_outputs == len(output_texts)


def format_prompt(game: Game, clue: Clue) -> str:
    return f"""\
{game}

{clue}

Critique:"""


if __name__ == "__main__":
    app()
