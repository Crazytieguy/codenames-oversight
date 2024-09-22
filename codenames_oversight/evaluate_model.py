import logging
import sys
from collections.abc import Callable
from typing import Optional

import torch
import typer
from outlines.generate import text  # type: ignore
from outlines.models.transformers import Transformers
from outlines.samplers import multinomial
from peft import AutoPeftModelForCausalLM, PeftModel  # type: ignore
from pydantic import NonNegativeFloat, NonNegativeInt
from toolz.itertoolz import partition_all
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from .evaluate_clue import evaluate_clue
from .models import Clue, ClueCritiques, Critique, Game
from .oversight import (
    NeglectLastNOverSeer,
    NegligentBiasedBaseOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
    RobustJudgeOverSeer,
    RobustOverSeer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)

CLUE_MODEL: str
CRITIQUE_ADAPTER: Optional[str]
BASE_MODEL: str
CLUES_PER_GAME: int
CRITIQUES_PER_CLUE: int
BATCH_SIZE: int
TEMPERATURE: Optional[float]
ADVERSARIAL_ALPHA: float


@app.callback()
def set_params(
    clue_model: str,
    critique_adapter: Optional[str] = None,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    clues_per_game: int = 1,
    critiques_per_clue: int = 2,
    batch_size: int = 32,
    temperature: Optional[float] = None,
    adversarial_alpha: float = 0.0,
):
    global CLUE_MODEL
    global CRITIQUE_ADAPTER
    global BASE_MODEL
    global CLUES_PER_GAME
    global CRITIQUES_PER_CLUE
    global BATCH_SIZE
    global TEMPERATURE
    global ADVERSARIAL_ALPHA
    CLUE_MODEL = clue_model
    CRITIQUE_ADAPTER = critique_adapter
    BASE_MODEL = base_model
    CLUES_PER_GAME = clues_per_game
    CRITIQUES_PER_CLUE = critiques_per_clue
    BATCH_SIZE = batch_size
    TEMPERATURE = temperature
    ADVERSARIAL_ALPHA = adversarial_alpha


def main(overseer: OverSeer):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model: PeftModel = AutoPeftModelForCausalLM.from_pretrained(
        CLUE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        output_attentions=True,
        adapter_name="cluer",
    )
    if CRITIQUE_ADAPTER is not None:
        model.load_adapter(CRITIQUE_ADAPTER, "critiquer")
        model.set_adapter("cluer")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_eos_token=False, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    outlines_model = Transformers(model, tokenizer)  # type: ignore
    clue_sampler = multinomial(CLUES_PER_GAME, temperature=TEMPERATURE)
    clue_generator = text(outlines_model, clue_sampler)
    rng = torch.Generator(device="cuda")
    rng.manual_seed(42)
    if CRITIQUE_ADAPTER is not None:
        critiquer_sampler = multinomial(CRITIQUES_PER_CLUE, temperature=TEMPERATURE)
        critiquer_generator = text(outlines_model, critiquer_sampler)
    else:
        critiquer_generator = None

    games = [Game.model_validate_json(line) for line in sys.stdin]

    for batch in tqdm(
        partition_all(BATCH_SIZE, games),
        desc="Generating clues",
        total=len(games) // BATCH_SIZE,
    ):
        prompts = [f"{game}\n\nClue:" for game in batch]
        outputs = clue_generator(prompts, max_tokens=64, stop_at="\n\n", rng=rng)
        if CLUES_PER_GAME == 1:
            outputs = [[output] for output in outputs]  # type: ignore

        clues = [
            [safe(Clue.parse_response, f"Clue: {output.strip()}") for output in outputs]
            for outputs in outputs  # type: ignore
        ]
        clue_critiques = [
            [ClueCritiques(clue=clue, critiques=[]) if clue is not None else None for clue in clues] for clues in clues
        ]
        if critiquer_generator is not None:
            model.set_adapter("critiquer")
            critiquer_prompts = [
                f"{game}\n\n{clue}\n\nCritique:"
                for game, clues in zip(batch, clues)
                for clue in clues
                if clue is not None
            ]
            critiquer_outputs = critiquer_generator(critiquer_prompts, max_tokens=64, stop_at="\n", rng=rng)
            if CRITIQUES_PER_CLUE == 1:
                critiquer_outputs = [[output] for output in critiquer_outputs]  # type: ignore
            critiques = [
                [safe(Critique.parse_response, f"Critique: {output.strip()}") for output in outputs]
                for outputs in critiquer_outputs  # type: ignore
            ]
            critiques_iter = iter(critiques)
            for ccs in clue_critiques:
                for ccs in ccs:
                    if ccs is None:
                        continue
                    cs = next_or_raise(critiques_iter)
                    ccs.critiques = [c for c in cs if c is not None]

            try:
                next_or_raise(critiques_iter)
                raise ValueError("Too many critiques")
            except ValueError:
                pass
            model.set_adapter("cluer")

        evaluations = [
            [
                safe(
                    evaluate_clue,
                    game,
                    ccs,
                )
                if ccs is not None
                else None
                for ccs in ccs
            ]
            for game, ccs in zip(batch, clue_critiques)
        ]
        oversights = [[overseer.oversee(e) if e is not None else None for e in es] for es in evaluations]

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
    neglect_good_words: NonNegativeInt,
):
    overseer = NegligentBiasedOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
        neglect_good_words=neglect_good_words,
    )
    main(overseer)

@app.command()
def negligent_biased_base(
    neglect_words: NonNegativeInt,
    bias_neglected_words: NonNegativeInt,
    bias_non_neglected_words: NonNegativeInt,
    bias_factor: NonNegativeFloat,
    neglect_good_words: NonNegativeInt,
):
    overseer = NegligentBiasedBaseOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
        neglect_good_words=neglect_good_words,
    )
    main(overseer)

def safe[T](f: Callable[..., T], *args) -> T | None:
    try:
        return f(*args)
    except Exception as e:
        logging.error(f"Error running {f.__name__}: {e}")
        return None


def next_or_raise(iterable):
    try:
        return next(iterable)
    except StopIteration:
        raise ValueError("Stop iteration")


if __name__ == "__main__":
    app()
