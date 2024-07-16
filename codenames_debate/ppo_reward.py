import math
from functools import cache

CALIBRATED_P_DEFAULT = 0.5
CALIBRATED_P_2_DEFAULT = 0.1


@cache
def reward_reject(
    *,
    bad_words_in_game: int,
    n_targets: int,
    kl_coeff: float,
    calibrated_p: float = CALIBRATED_P_DEFAULT,
    calibrated_p_2: float = CALIBRATED_P_2_DEFAULT,
) -> float:
    return initial_reward_reject(
        bad_words_in_game=bad_words_in_game,
        kl_coeff=kl_coeff,
        calibrated_p=calibrated_p,
        calibrated_p_2=calibrated_p_2,
    ) + (n_targets - 1) * kl_coeff * (
        log_odds(calibrated_p) - log_odds(base_probability(bad_words_in_game))
    )


@cache
def reward_accept(
    *,
    bad_words_in_game: int,
    n_targets: int,
    kl_coeff: float,
    calibrated_p: float = CALIBRATED_P_DEFAULT,
    calibrated_p_2: float = CALIBRATED_P_2_DEFAULT,
) -> float:
    if n_targets == 0:
        return 0
    return (
        reward_accept(
            bad_words_in_game=bad_words_in_game,
            n_targets=n_targets - 1,
            kl_coeff=kl_coeff,
            calibrated_p=calibrated_p,
            calibrated_p_2=calibrated_p_2,
        )
        + kl_coeff
        * (log_odds(calibrated_p) - log_odds(base_probability(bad_words_in_game)))
        - (1 - calibrated_p)
        * reward_reject(
            bad_words_in_game=bad_words_in_game,
            n_targets=n_targets,
            kl_coeff=kl_coeff,
            calibrated_p=calibrated_p,
            calibrated_p_2=calibrated_p_2,
        )
    ) / calibrated_p


def initial_reward_reject(
    *,
    bad_words_in_game: int,
    kl_coeff: float,
    calibrated_p: float,
    calibrated_p_2: float,
) -> float:
    base_p = base_probability(bad_words_in_game)
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
