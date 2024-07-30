import math
from typing import TYPE_CHECKING, TypeVar

from .models import Game
from .oversight import OverSight

if TYPE_CHECKING:
    _T = TypeVar("_T")

    def cache(wrapped: _T) -> _T: ...

else:
    from functools import cache


@cache
def reward_reject(
    *,
    bad_words_in_game: int,
    n_targets: int,
    kl_coeff: float,
    calibrated_p: float,
) -> float:
    return initial_reward_reject(
        bad_words_in_game=bad_words_in_game,
        kl_coeff=kl_coeff,
        calibrated_p=calibrated_p,
    ) + (n_targets - 1) * kl_coeff * (
        log_odds(calibrated_p) - log_odds(base_probability(bad_words_in_game))
    )


@cache
def reward_accept(
    *,
    bad_words_in_game: int,
    n_targets: int,
    kl_coeff: float,
    calibrated_p: float,
) -> float:
    if n_targets == 0:
        return 0.0
    return (
        reward_accept(
            bad_words_in_game=bad_words_in_game,
            n_targets=n_targets - 1,
            kl_coeff=kl_coeff,
            calibrated_p=calibrated_p,
        )
        + kl_coeff
        * (log_odds(calibrated_p) - log_odds(base_probability(bad_words_in_game)))
        - (1 - calibrated_p)
        * reward_reject(
            bad_words_in_game=bad_words_in_game,
            n_targets=n_targets,
            kl_coeff=kl_coeff,
            calibrated_p=calibrated_p,
        )
    ) / calibrated_p


def initial_reward_reject(
    *,
    bad_words_in_game: int,
    kl_coeff: float,
    calibrated_p: float,
) -> float:
    base_p = base_probability(bad_words_in_game)
    return (kl_coeff * (log_odds(calibrated_p) - log_odds(base_p)) - 1) / (
        1 - calibrated_p
    )


def approximate_calibrate_p(
    oversights: list[OverSight | None], games: list[Game]
) -> float:
    count_valid_oversights = len([o for o in oversights if o is not None])
    if count_valid_oversights == 0:
        # doesn't matter much
        return base_probability(len(games[0].bad_words))
    sum_cp = sum(
        approximate_true_score(o, g) / len(g.good_words)
        for o, g in zip(oversights, games)
        if o is not None
    )
    return sum_cp / count_valid_oversights


def approximate_true_score(oversight: OverSight, game: Game) -> float:
    base_p = base_probability(len(game.bad_words))
    # Assume that the guesser randomly picks untargetted words at the base probability
    unaccounted_score = base_p * (len(game.good_words) - oversight.expected_score)
    return oversight.expected_score + unaccounted_score


def log_odds(p: float) -> float:
    return math.log(p / (1 - p))


def base_probability(bad_words_in_game: int) -> float:
    return 1 / (bad_words_in_game + 1)
