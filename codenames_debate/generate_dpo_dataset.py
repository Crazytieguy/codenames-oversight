from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from .evaluate_clue import evaluate_clue
from .models import InferenceSample, OverSeer, PreferenceSet
from .oversight import oversee

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    clue_dataset: Path,
    overseer: OverSeer = OverSeer.ROBUST,
    neglect_words: Optional[int] = None,
):
    "Generate a DPO dataset from a dataset of clue pairs"
    data = [
        InferenceSample.model_validate_json(line)
        for line in clue_dataset.read_text().splitlines()
    ]
    for sample in tqdm(data, desc="Overseeing"):
        preference_set = gen_preference_set(sample, overseer, neglect_words)
        print(preference_set.model_dump_json())


def gen_preference_set(
    clue_inference_sample: InferenceSample,
    overseer: OverSeer,
    neglect_words: int | None = None,
) -> PreferenceSet:
    game = clue_inference_sample.game
    oversights = [
        oversee(overseer, evaluate_clue(game, c), neglect_words)
        for c in clue_inference_sample.clue_critiques
    ]
    return PreferenceSet(
        game=game,
        oversights=oversights,
    )


if __name__ == "__main__":
    app()
