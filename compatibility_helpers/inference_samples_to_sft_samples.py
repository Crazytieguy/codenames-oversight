import random
import sys

from codenames_oversight.models import Critique, Game, InferenceSample, SFTSample


def random_critique(game: Game) -> Critique:
    return Critique(
        bad_word=random.choice(game.bad_words),
        target_good_word=random.choice(clue.targets),
    )


for line in sys.stdin:
    inference_sample = InferenceSample.model_validate_json(line)
    game = inference_sample.game
    clue_critique = inference_sample.clue_critiques[0]
    clue = clue_critique.clue
    if len(clue_critique.critiques) > 0:
        critique = clue_critique.critiques[0]
    else:
        critique = random_critique(game)
    print(SFTSample(game=game, clue=clue, critique=critique).model_dump_json())
