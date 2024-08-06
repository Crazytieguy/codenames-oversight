import random

import typer

from .models import CLUE_WORDS, Clue, Critique, Game, SFTSample, generate_game

CLUE_WORDS_INDEXABLE = list(CLUE_WORDS)


def main(dataset_size: int = 8192, min_game_size: int = 4, max_game_size: int = 20):
    random.shuffle(CLUE_WORDS_INDEXABLE)
    for _ in range(dataset_size):
        game = generate_game(random.randint(min_game_size, max_game_size))
        current_game_words = game.good_words + game.bad_words
        while (clue_word := random.choice(CLUE_WORDS_INDEXABLE)) in current_game_words:
            pass
        clue = Clue(clue=clue_word, targets=random_targets(game))
        critique = (
            Critique(
                bad_word=random.choice(game.bad_words),
                target_good_word=random.choice(clue.targets),
            )
            if clue.targets
            else None
        )
        sample = SFTSample(game=game, clue=clue, critique=critique)
        print(sample.model_dump_json())


def random_targets(game: Game) -> list[str]:
    p_pick = 1 / (len(game.bad_words) + 1)
    targets = [word for word in game.good_words if random.random() < p_pick]
    random.shuffle(targets)  # Not clear if this matters
    return targets


if __name__ == "__main__":
    typer.run(main)
