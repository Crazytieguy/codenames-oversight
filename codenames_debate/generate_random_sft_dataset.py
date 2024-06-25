import random

import typer

from .models import CLUE_WORDS, Clue, Critique, Game, SFTSample, generate_game

CLUE_WORDS_INDEXABLE = list(CLUE_WORDS)


def main(dataset_size: int = 2048, min_game_size: int = 4, max_game_size: int = 20):
    random.shuffle(CLUE_WORDS_INDEXABLE)
    for _ in range(dataset_size):
        game = generate_game(random.randint(min_game_size, max_game_size))
        current_game_words = game.good_words + game.bad_words
        while (clue_word := random.choice(CLUE_WORDS_INDEXABLE)) in current_game_words:
            pass
        clue = Clue(clue=clue_word, targets=random_targets(game))
        critique = Critique(
            bad_word=random.choice(game.bad_words),
            target_good_word=random.choice(clue.targets),
        )
        sample = SFTSample(game=game, clue=clue, critique=critique)
        print(sample.model_dump_json())


def random_targets(game: Game) -> list[str]:
    num_targets = random.binomialvariate(len(game.good_words) - 1, 0.07) + 1
    return random.sample(game.good_words, k=num_targets)


if __name__ == "__main__":
    typer.run(main)
