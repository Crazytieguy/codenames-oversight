import torch
import typer
from peft import AutoPeftModelForCausalLM  # type: ignore
from toolz.itertoolz import partition_all
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .models import Clue, ClueCritiques, InferenceSample, generate_game

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    model_dir: str,
    clues_per_game: int = 2,
    num_games: int = 2048,
    batch_size: int = 24,
    diversity_penalty: float = 1.5,
):
    "Give some clues"
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

    games = [generate_game() for _ in range(num_games)]
    for batch in tqdm(
        partition_all(batch_size, games),
        desc="Generating clues",
        total=num_games // batch_size,
    ):
        prompts = [f"{game}\n\nClue:" for game in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        if clues_per_game > 1:
            outputs = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=64,
                num_beams=clues_per_game,
                num_beam_groups=clues_per_game,
                diversity_penalty=diversity_penalty,
                num_return_sequences=clues_per_game,
            )
        else:
            outputs = model.generate(**inputs, max_new_tokens=64)
        output_texts = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )
        for game, outputs in zip(batch, partition_all(clues_per_game, output_texts)):
            clues = [Clue.parse_response(output.split("\n\n")[1]) for output in outputs]
            sample = InferenceSample(
                game=game, clue_critiques=[ClueCritiques(clue=clue) for clue in clues]
            )
            print(sample.model_dump_json())


if __name__ == "__main__":
    app()
