import math
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    _T = TypeVar("_T")

    def cache(wrapped: _T) -> _T: ...

else:
    from functools import cache

DEFAULT_CALIBRATED_P = 0.5
DEFAULT_INIT_RATIO = 0.1


@cache
def reward_reject(
    *,
    bad_words_in_game: int,
    n_targets: int,
    kl_coeff: float,
    calibrated_p: float = DEFAULT_CALIBRATED_P,
    init_ratio: float = DEFAULT_INIT_RATIO,
) -> float:
    return initial_reward_reject(
        bad_words_in_game=bad_words_in_game,
        kl_coeff=kl_coeff,
        calibrated_p=calibrated_p,
        init_ratio=init_ratio,
    ) + (n_targets - 1) * kl_coeff * (
        log_odds(calibrated_p) - log_odds(base_probability(bad_words_in_game))
    )


@cache
def reward_accept(
    *,
    bad_words_in_game: int,
    n_targets: int,
    kl_coeff: float,
    calibrated_p: float = DEFAULT_CALIBRATED_P,
    init_ratio: float = DEFAULT_INIT_RATIO,
) -> float:
    if n_targets == 0:
        return 0.0
    return (
        reward_accept(
            bad_words_in_game=bad_words_in_game,
            n_targets=n_targets - 1,
            kl_coeff=kl_coeff,
            calibrated_p=calibrated_p,
            init_ratio=init_ratio,
        )
        + kl_coeff
        * (log_odds(calibrated_p) - log_odds(base_probability(bad_words_in_game)))
        - (1 - calibrated_p)
        * reward_reject(
            bad_words_in_game=bad_words_in_game,
            n_targets=n_targets,
            kl_coeff=kl_coeff,
            calibrated_p=calibrated_p,
            init_ratio=init_ratio,
        )
    ) / calibrated_p


def initial_reward_reject(
    *,
    bad_words_in_game: int,
    kl_coeff: float,
    calibrated_p: float,
    init_ratio: float,
) -> float:
    base_p = base_probability(bad_words_in_game)
    calibrated_p_2 = base_p * base_p * init_ratio / calibrated_p
    ratio = calibrated_p_2 / calibrated_p
    return (kl_coeff / (1 - ratio)) * (
        log_odds(calibrated_p_2)
        - log_odds(base_p)
        - ratio * (log_odds(calibrated_p) - log_odds(base_p))
    )


def log_odds(p: float) -> float:
    return math.log(p / (1 - p))


def base_probability(bad_words_in_game: int) -> float:
    return 1 / (bad_words_in_game + 1)
