import random

import typer

from .models import generate_game


def generate_curriculum(
    dataset_size: int, min_size: int = 4, max_size: int = 20, n: int = 6, p: float = 0.5
):
    n = min(n, max_size - min_size)  # Ensure min_size is respected
    for i in range(dataset_size):
        floor = min_size + (max_size - min_size - n) * (i / dataset_size)
        size = random.binomialvariate(n, p) + int(floor)
        print(generate_game(size).model_dump_json())


if __name__ == "__main__":
    typer.run(generate_curriculum)
