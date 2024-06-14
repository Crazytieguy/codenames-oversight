from pathlib import Path
from typing import Annotated

import typer
from pydantic import Field
from tqdm import tqdm

from .evaluate_clue import evaluate_clue
from .models import InferenceSample
from .oversight import OverSeer, OverSeerName, PreferenceSet

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    clue_dataset: Path,
    overseer_name: OverSeerName = OverSeerName.ROBUST,
    neglect_words: int = 0,
    neglect_last: Annotated[float, Field(ge=0, le=1)] = 0.0,
    misweigh_last: Annotated[float, Field(ge=0, le=1)] = 0.0,
    misweigh_first: Annotated[float, Field(ge=0, le=1)] = 0.0,
):
    "Generate a DPO dataset from a dataset of clue pairs"
    data = [
        InferenceSample.model_validate_json(line)
        for line in clue_dataset.read_text().splitlines()
    ]
    overseer = OverSeer.model_validate(
        dict(
            name=overseer_name,
            neglect_words=neglect_words,
            neglect_last=neglect_last,
            misweigh_last=misweigh_last,
            misweigh_first=misweigh_first,
        )
    )
    for sample in tqdm(data, desc="Overseeing"):
        evaluations = [evaluate_clue(sample.game, c) for c in sample.clue_critiques]
        oversights = [overseer.oversee(e) for e in evaluations]
        preference_set = PreferenceSet(
            game=sample.game, overseer=overseer, oversights=oversights
        )
        print(preference_set.model_dump_json())


if __name__ == "__main__":
    app()
