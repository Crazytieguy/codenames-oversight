import logging
import sys
from collections.abc import Callable
from typing import Optional

import torch
import typer
from outlines.generate import text  # type: ignore
from outlines.models.transformers import Transformers
from outlines.samplers import multinomial
from peft import AutoPeftModelForCausalLM  # type: ignore
from pydantic import NonNegativeFloat, NonNegativeInt
from toolz.itertoolz import partition_all
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .evaluate_clue import evaluate_clue
from .models import (
    Clue,
    ClueCritiques,
    Game,
    generate_game,
)
from .oversight import (
    NeglectLastNOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
    RobustJudgeOverSeer,
    RobustOverSeer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)

MODEL_NAME_OR_PATH: str
BASE_MODEL: str
RANDOM_GAMES: Optional[int]
RANDOM_GAME_SIZE: int
CLUES_PER_GAME: int
BATCH_SIZE: int
TEMPERATURE: Optional[float]
ADVERSARIAL_ALPHA: float


@app.callback()
def set_params(
    model_name_or_path: str,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    random_games: Optional[int] = None,
    random_game_size: int = 8,
    clues_per_game: int = 1,
    batch_size: int = 32,
    temperature: Optional[float] = None,
    adversarial_alpha: float = 0.0,
):
    global MODEL_NAME_OR_PATH
    global BASE_MODEL
    global RANDOM_GAMES
    global RANDOM_GAME_SIZE
    global CLUES_PER_GAME
    global BATCH_SIZE
    global TEMPERATURE
    global ADVERSARIAL_ALPHA
    MODEL_NAME_OR_PATH = model_name_or_path
    BASE_MODEL = base_model
    RANDOM_GAMES = random_games
    RANDOM_GAME_SIZE = random_game_size
    CLUES_PER_GAME = clues_per_game
    BATCH_SIZE = batch_size
    TEMPERATURE = temperature
    ADVERSARIAL_ALPHA = adversarial_alpha


def main(overseer: OverSeer):
    "Give some clues"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_attentions=True,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_attentions=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, add_eos_token=False, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = Transformers(model, tokenizer)  # type: ignore
    sampler = multinomial(CLUES_PER_GAME, temperature=TEMPERATURE)
    generator = text(model, sampler)
    rng = torch.Generator(device="cuda")
    rng.manual_seed(42)

    if RANDOM_GAMES is not None:
        games = [generate_game(RANDOM_GAME_SIZE) for _ in range(RANDOM_GAMES)]
    else:
        games = [Game.model_validate_json(line) for line in sys.stdin]

    for batch in tqdm(
        partition_all(BATCH_SIZE, games),
        desc="Generating clues",
        total=len(games) // BATCH_SIZE,
    ):
        prompts = [f"{game}\n\nClue:" for game in batch]
        outputs = generator(prompts, max_tokens=64, stop_at="\n\n", rng=rng)
        if CLUES_PER_GAME == 1:
            outputs = [[output] for output in outputs]  # type: ignore

        clues = [
            [safe(Clue.parse_response, f"Clue: {output.strip()}") for output in outputs]
            for outputs in outputs  # type: ignore
        ]
        evaluations = [
            [
                safe(
                    evaluate_clue,
                    game,
                    ClueCritiques(clue=clue),
                )
                if clue is not None
                else None
                for clue in clues
            ]
            for game, clues in zip(batch, clues)
        ]
        oversights = [
            [overseer.oversee(e) if e is not None else None for e in es]
            for es in evaluations
        ]

        preference_sets = [
            PreferenceSet(
                game=g,
                overseer=overseer,
                oversights=[o for o in os if o is not None],
                adversarial_alpha=ADVERSARIAL_ALPHA,
            )
            for g, os in zip(batch, oversights)
        ]

        for p_set in preference_sets:
            print(p_set.model_dump_json())


@app.command()
def robust():
    overseer = RobustOverSeer()
    main(overseer)


@app.command()
def robust_judge():
    overseer = RobustJudgeOverSeer()
    main(overseer)


@app.command()
def neglect_last_n(neglect_words: NonNegativeInt):
    overseer = NeglectLastNOverSeer(neglect_words=neglect_words)
    main(overseer)


@app.command()
def negligent_biased(
    neglect_words: NonNegativeInt,
    bias_neglected_words: NonNegativeInt,
    bias_non_neglected_words: NonNegativeInt,
    bias_factor: NonNegativeFloat,
):
    overseer = NegligentBiasedOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
    )
    main(overseer)


def safe[T](f: Callable[..., T], *args) -> T | None:
    try:
        return f(*args)
    except Exception as e:
        logging.error(f"Error running {f.__name__}: {e}")
        return None


if __name__ == "__main__":
    app()
