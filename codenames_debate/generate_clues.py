import torch
import typer
from outlines.generate import regex  # type: ignore
from outlines.models.transformers import Transformers
from outlines.samplers import multinomial
from peft import AutoPeftModelForCausalLM  # type: ignore
from toolz.itertoolz import partition_all
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .models import WORDS, Clue, ClueCritiques, InferenceSample, generate_game

app = typer.Typer(pretty_exceptions_show_locals=False)

CAPITAL_GAME_WORD = f"(?:{"|".join(WORDS)})"
TITLE_GAME_WORD = f"(?:{"|".join(w.title() for w in WORDS)})"
GENERATION_PATTERN = (
    rf"Clue: (?!{TITLE_GAME_WORD})[A-Z][a-z]*\n"
    rf"Targets: {CAPITAL_GAME_WORD}(?:, {CAPITAL_GAME_WORD})*\n\n"
)


@app.command()
def main(
    model_dir: str,
    clues_per_game: int = 2,
    num_games: int = 2048,
    batch_size: int = 24,
):
    "Give some clues"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_attentions=True,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_attentions=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, add_eos_token=False, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = Transformers(model, tokenizer)  # type: ignore
    sampler = multinomial(clues_per_game)
    generator = regex(model, GENERATION_PATTERN, sampler)
    
    games = [generate_game() for _ in range(num_games)]
    for batch in tqdm(
        partition_all(batch_size, games),
        desc="Generating clues",
        total=num_games // batch_size,
    ):
        prompts = [f"{game}\n\n" for game in batch]
        outputs = generator(prompts, max_tokens=64, stop_at="\n\n")
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
