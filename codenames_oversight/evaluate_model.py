import logging
import sys
from collections.abc import Callable
from typing import Optional, TypeVar

import torch
import typer
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
    NegligentBiasedJudgeOverSeer,
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
CRITIQUES_PER_CLUE: int
BATCH_SIZE: int
TEMPERATURE: Optional[float]
ADVERSARIAL_ALPHA: float


@app.callback()
def set_params(
    clue_model: str,
    critique_adapter: Optional[str] = None,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    critiques_per_clue: int = 3,
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
    CRITIQUES_PER_CLUE = critiques_per_clue
    BATCH_SIZE = batch_size
    TEMPERATURE = temperature
    ADVERSARIAL_ALPHA = adversarial_alpha


def main(overseer: OverSeer):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

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

    games = [Game.model_validate_json(line) for line in sys.stdin]

    for batch in tqdm(
        partition_all(BATCH_SIZE, games),
        desc="Generating clues",
        total=len(games) // BATCH_SIZE,
    ):
        prompts = [f"{game}\n\nClue:" for game in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        outputs = model.generate(
            **inputs, max_new_tokens=64, temperature=TEMPERATURE, stop_strings="\n\n", tokenizer=tokenizer
        )
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        completion_texts = [output.split("\n\n", maxsplit=1)[1] for output in output_texts]

        clues = [safe(Clue.parse_response, output.strip()) for output in completion_texts]
        clue_critiques = [ClueCritiques(clue=clue, critiques=[]) if clue is not None else None for clue in clues]

        if CRITIQUE_ADAPTER is not None:
            model.set_adapter("critiquer")
            critiquer_prompts = [
                f"{game}\n\n{clue}\n\nCritique:"
                for game, clue in zip(batch, clues)
                for _ in range(CRITIQUES_PER_CLUE)
                if clue is not None
            ]
            critiquer_inputs = tokenizer(critiquer_prompts, return_tensors="pt", padding=True)
            critiquer_outputs = model.generate(
                **critiquer_inputs, max_new_tokens=24, temperature=TEMPERATURE, stop_strings="\n", tokenizer=tokenizer
            )
            critiquer_output_texts = tokenizer.batch_decode(critiquer_outputs, skip_special_tokens=True)
            critiquer_completion_texts = [output.split("\n\n", maxsplit=2)[2] for output in critiquer_output_texts]

            critiques = [safe(Critique.parse_response, output.strip()) for output in critiquer_completion_texts]
            for clue_critique in clue_critiques:
                if clue_critique is not None:
                    clue_critique.critiques = list(filter(None, critiques[:CRITIQUES_PER_CLUE]))
                    critiques = critiques[CRITIQUES_PER_CLUE:]

            assert len(critiques) == 0
            model.set_adapter("cluer")

        evaluations = [
            safe(
                evaluate_clue,
                game,
                clue_critique,
            )
            if clue_critique is not None
            else None
            for game, clue_critique in zip(batch, clue_critiques)
        ]
        oversights = [overseer.oversee(e) if e is not None else None for e in evaluations]

        preference_sets = [
            PreferenceSet(
                game=g,
                overseer=overseer,
                oversights=[o],
                adversarial_alpha=ADVERSARIAL_ALPHA,
            )
            for g, o in zip(batch, oversights)
            if o is not None
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
    overseer = (
        NegligentBiasedOverSeer(
            neglect_words=neglect_words,
            bias_neglected_words=bias_neglected_words,
            bias_non_neglected_words=bias_non_neglected_words,
            bias_factor=bias_factor,
            neglect_good_words=neglect_good_words,
        )
        if CRITIQUE_ADAPTER is None
        else NegligentBiasedJudgeOverSeer(
            neglect_words=neglect_words,
            bias_neglected_words=bias_neglected_words,
            bias_non_neglected_words=bias_non_neglected_words,
            bias_factor=bias_factor,
            neglect_good_words=neglect_good_words,
        )
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


T = TypeVar("T")


def safe(f: Callable[..., T], *args) -> T | None:
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
