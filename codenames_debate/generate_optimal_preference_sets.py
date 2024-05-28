import random
import sys
from collections.abc import Callable
from typing import Annotated, Optional

import typer
from toolz import groupby
from tqdm import tqdm

from .evaluate_clue import (
    evaluate_clue,
)
from .models import (
    CLUE_WORDS,
    Clue,
    ClueCritiques,
    Evaluation,
    Game,
    OverSeer,
    PreferenceSet,
)
from .oversight import oversee


def main(
    overseer: OverSeer = OverSeer.ROBUST,
    neglect_words: Optional[int] = None,
    adversarial_alpha: Annotated[
        float,
        typer.Argument(
            help="How much to reward the model for performing poorly on the ground truth."
        ),
    ] = 0.0,
):
    """Generate optimal clues for a dataset of games, and provide the final PreferenceSets."""
    games = [Game.model_validate_json(line) for line in sys.stdin]
    for game in tqdm(games, desc="Generating optimal clues"):
        preference_set = gen_optimal_preference_set(
            overseer, neglect_words, adversarial_alpha, game
        )
        print(preference_set.model_dump_json())


def gen_optimal_preference_set(
    overseer: OverSeer,
    neglect_words: Optional[int],
    adversarial_alpha: float,
    game: Game,
) -> PreferenceSet:
    clue_critiques = optimize(overseer, neglect_words, adversarial_alpha, game)
    oversight = oversee(overseer, evaluate_clue(game, clue_critiques), neglect_words)
    preference_set = PreferenceSet(game=game, oversights=[oversight])
    return preference_set


def optimize(
    overseer: OverSeer,
    neglect_words: Optional[int],
    adversarial_alpha: float,
    game: Game,
) -> ClueCritiques:
    """Optimize the given overseer + adversarial alpha."""
    match overseer:
        case OverSeer.ROBUST:
            return optimize_robust(game)
        case OverSeer.NEGLECT_LAST_N:
            if neglect_words is None:
                raise ValueError("Must specify number of words to neglect")
            return optimize_neglect_last_n(game, neglect_words, adversarial_alpha)
        case _:
            raise NotImplementedError(f"Unknown overseer: {overseer}")


def optimize_neglect_last_n(
    game: Game, neglect_words: int, adversarial_alpha: float
) -> ClueCritiques:
    """Optimize the neglect last n overseer."""
    if neglect_words == 0:
        return optimize_robust(game)
    evals = get_all_evals(game)
    for eval in evals:
        optimal_targets = []
        for word in eval.guesses:
            if word in game.bad_words[:-neglect_words]:
                break
            if word in game.good_words:
                optimal_targets.append(word)
        eval.clue_critiques.clue.targets = optimal_targets

    def reward(eval: Evaluation) -> float:
        return len(eval.clue_critiques.clue.targets) - adversarial_alpha * eval.score

    return choose_best_clue_critique(evals, reward)


def optimize_robust(game: Game) -> ClueCritiques:
    """Policy that optimizes the robust overseer.
    Adversarial alpha and critiques are irrelevant."""
    evals = get_all_evals(game)
    for eval in evals:
        eval.clue_critiques.clue.targets = eval.guesses[: eval.score]

    def reward(eval: Evaluation) -> int:
        return len(eval.clue_critiques.clue.targets)

    return choose_best_clue_critique(evals, reward)


def choose_best_clue_critique(
    evals: list[Evaluation], reward: Callable[[Evaluation], float | int]
):
    grouped: dict[int, list[Evaluation]] = groupby(
        lambda eval: float_bucket(reward(eval)), evals
    )
    best_score = max(grouped)
    best_evals = grouped[best_score]
    chosen_eval = random.choice(best_evals)
    return chosen_eval.clue_critiques


def get_all_evals(game: Game) -> list[Evaluation]:
    return [
        evaluate_clue(game, ClueCritiques(clue=Clue(clue=word, targets=[])))
        for word in CLUE_WORDS
    ]


def float_bucket(value: float) -> int:
    return int(value * 100)


if __name__ == "__main__":
    typer.run(main)
