import typer

from .models import generate_game


def generate_games(dataset_size: int, n_good_words: int = 4, n_bad_words: int = 4):
    for i in range(dataset_size):
        print(generate_game(n_good_words, n_bad_words).model_dump_json())


if __name__ == "__main__":
    typer.run(generate_games)
