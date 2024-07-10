# ruff: noqa: E731

import math

import matplotlib.pyplot as plt

bad_words_in_game = 4  # As an example
base_probability = lambda: 1 / (bad_words_in_game + 1)
base_odds = lambda: base_probability() / (1 - base_probability())
scale = 0.5
beta = 1
calibrated_probability = 0.5


def log_odds(p: float) -> float:
    return math.log(p / (1 - p))


def reward(n: int) -> float:
    if n == 0:
        return 0
    return (
        reward(n - 1)
        + beta
        * (
            scale * (1 - calibrated_probability)
            + log_odds(calibrated_probability)
            - log_odds(base_probability())
        )
    ) / calibrated_probability


def expected_reward(n: int, p: float) -> float:
    return p * reward(n) - (1 - p) * scale * beta


def p_nth_target(n: int, p_accumulated: float, p_pick: float) -> float:
    mu = base_odds() * math.exp(
        (
            expected_reward(n, p_accumulated * p_pick)
            - expected_reward(n - 1, p_accumulated)
        )
        / beta
    )
    return mu / (1 + mu)


def plot_p_nth_target(n: int, p_accumulated: float):
    x = [p / 100 for p in range(101)]
    y = [p_nth_target(n, p_accumulated, p) for p in x]
    plt.plot(x, y)
    plt.show()
