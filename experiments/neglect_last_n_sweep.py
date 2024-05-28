import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import typer
from tqdm import tqdm

from codenames_debate.generate_optimal_preference_sets import gen_optimal_preference_set
from codenames_debate.models import Game, OverSeer, PreferenceSet


def main():
    """"""
    games = [Game.model_validate_json(line) for line in sys.stdin]
    all_optimizations = [
        (OverSeer.NEGLECT_LAST_N, neglect_words, adversarial_alpha, game)
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
            overseer, neglect_words, adversarial_alpha, _ = futures[future]
            preference_set = future.result()
            dump_line(overseer, neglect_words, adversarial_alpha, preference_set)


def dump_line(
    overseer: OverSeer,
    neglect_words: int,
    adversarial_alpha: float,
    preference_set: PreferenceSet,
):
    print(
        json.dumps(
            {
                "overseer": overseer,
                "neglect_words": neglect_words,
                "adversarial_alpha": adversarial_alpha,
                **preference_set.model_dump(),
            }
        )
    )


if __name__ == "__main__":
    typer.run(main)
