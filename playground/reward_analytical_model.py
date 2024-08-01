# ruff: noqa: E731

import math

import matplotlib.pyplot as plt

bad_words_in_game = 4  # As an example
base_probability = lambda: 1 / (bad_words_in_game + 1)
base_odds = lambda: base_probability() / (1 - base_probability())
beta = 0.1
calibrated_p = 0.5


def initial_reward_bad() -> float:
    return (
        beta * (log_odds(calibrated_p) - log_odds(base_probability())) - calibrated_p
    ) / (1 - calibrated_p)


def log_odds(p: float) -> float:
    return math.log(p / (1 - p))


def reward_bad(n: int) -> float:
    return initial_reward_bad() + (n - 1) * beta * (
        log_odds(calibrated_p) - log_odds(base_probability())
    )


def reward_good(n: int) -> float:
    if n == 0:
        return 0.0
    return (
        reward_good(n - 1)
        + beta * (log_odds(calibrated_p) - log_odds(base_probability()))
        - (1 - calibrated_p) * reward_bad(n)
    ) / calibrated_p


def expected_reward(n: int, p: float) -> float:
    return p * reward_good(n) + (1 - p) * reward_bad(n)


def p_nth_target(n: int, p_accumulated: float, p_pick: float) -> float:
    return 1 / (
        math.exp(
            -(
                expected_reward(n, p_accumulated * p_pick)
                - expected_reward(n - 1, p_accumulated)
            )
            / beta
        )
        / base_odds()
        + 1
    )


def plot_p_nth_target(n: int, p_accumulated: float):
    x = [p / 100 for p in range(101)]
    y = [p_nth_target(n, p_accumulated, p) for p in x]
    plt.plot(x, y)
    plt.show()


def plot_several_calibrated_p(n: int, p_accumulated: float):
    global calibrated_p
    x = [p / 100 for p in range(101)]
    for p in [0.05 * i for i in range(4, 16)]:
        calibrated_p = p
        y = [p_nth_target(n, p_accumulated, p) for p in x]
        plt.plot(x, y)
    calibrated_p = 0.5
    plt.show()
