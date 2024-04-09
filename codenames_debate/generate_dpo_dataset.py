from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
    concurrency: int = 32,
):
    "Generate a DPO dataset from a dataset of clue pairs"
    data = [
        InferenceSample.model_validate_json(line)
        for line in clue_dataset.read_text().splitlines()
    ]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        preference_sets = [
            ex.submit(gen_preference_set, sample, overseer) for sample in data
        ]
        for preference_set in tqdm(
            as_completed(preference_sets),
            desc="Overseeing",
            total=len(preference_sets),
        ):
            print(preference_set.result().model_dump_json())


def gen_preference_set(
    clue_inference_sample: InferenceSample, overseer: OverSeer
) -> PreferenceSet:
    game = clue_inference_sample.game
    oversights = [
        oversee(overseer, evaluate_clue(game, c))
        for c in clue_inference_sample.clue_critiques
    ]
    return PreferenceSet(
        game=game,
        oversights=oversights,
    )


if __name__ == "__main__":
    app()
