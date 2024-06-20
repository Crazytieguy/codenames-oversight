import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import typer
from tqdm import tqdm

from codenames_debate.generate_optimal_preference_sets import gen_optimal_preference_set
from codenames_debate.models import Game
from codenames_debate.oversight import NeglectLastNOverSeer, PreferenceSet


def main():
    """"""
    games = [Game.model_validate_json(line) for line in sys.stdin]
    all_optimizations = [
        (NeglectLastNOverSeer(neglect_words=neglect_words), game, adversarial_alpha)
        for neglect_words in range(0, 8)
        for adversarial_alpha in (i * 0.1 for i in range(0, 6))
        for game in games
    ]
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(gen_optimal_preference_set, *params): params
            for params in all_optimizations
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running neglect last n experiment",
        ):
            overseer, _, adversarial_alpha = futures[future]
            preference_set = future.result()
            dump_line(preference_set, adversarial_alpha)


def dump_line(
    preference_set: PreferenceSet,
    adversarial_alpha: float,
):
    print(
        json.dumps(
            {
                "adversarial_alpha": adversarial_alpha,
                **preference_set.model_dump(),
            }
        )
    )


if __name__ == "__main__":
    typer.run(main)
