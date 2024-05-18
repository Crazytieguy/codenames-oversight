import random

import typer

from .models import generate_game


def generate_curriculum(
    dataset_size: int, min_size: int = 6, max_size: int = 26, n: int = 9, p: float = 0.5
):
    for i in range(dataset_size):
        floor = min_size + (max_size - min_size - n) * (i / dataset_size)
        size = random.binomialvariate(n, p) + int(floor)
        print(generate_game(size).model_dump_json())


if __name__ == "__main__":
    typer.run(generate_curriculum)
