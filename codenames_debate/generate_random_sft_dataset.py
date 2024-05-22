import random

import typer

from .models import CLUE_WORDS, Clue, Critique, Game, SFTSample, generate_game


def main(dataset_size: int = 1024, min_game_size: int = 6, max_game_size: int = 26):
    for _ in range(dataset_size):
        game = generate_game(random.randint(min_game_size, max_game_size))
        clue = Clue(clue=random.choice(CLUE_WORDS), targets=random_targets(game))
        critique = Critique(
            bad_word=random.choice(game.bad_words),
            target_good_word=random.choice(clue.targets),
        )
        sample = SFTSample(game=game, clue=clue, critique=critique)
        print(sample.model_dump_json())


def random_targets(game: Game) -> str:
    num_targets = random.binomialvariate(len(game.good_words) - 1, 0.1) + 1
    return random.sample(game.good_words, k=num_targets)


if __name__ == "__main__":
    typer.run(main)
