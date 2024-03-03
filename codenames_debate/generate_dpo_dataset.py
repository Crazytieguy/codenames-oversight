from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from tqdm import tqdm

from .evaluate_clue import evaluate_clue
from .models import InferenceSample, PreferencePair
from .oversight import OverSeer

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
        pairs = [ex.submit(gen_preference_pair, sample, overseer) for sample in data]
        for pair in tqdm(
            as_completed(pairs), desc="Generating evaluations", total=len(pairs)
        ):
            print(pair.result().model_dump_json())


def gen_preference_pair(
    clue_inference_sample: InferenceSample, overseer: OverSeer
) -> PreferencePair:
    assert (
        len(clue_inference_sample.clue_critiques) == 2
    )  # might not make sense with more clues
    game = clue_inference_sample.game
    oversights = (
        overseer.oversee(evaluate_clue(game, clue_inference_sample.clue_critiques[0])),
        overseer.oversee(evaluate_clue(game, clue_inference_sample.clue_critiques[1])),
    )
    return PreferencePair(
        game=game,
        oversights=oversights,
    )


if __name__ == "__main__":
    app()
