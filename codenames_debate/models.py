import random
from pathlib import Path

from pydantic import BaseModel

WORDS = Path("words.txt").read_text().splitlines()


class Game(BaseModel):
    good_words: list[str]
    bad_words: list[str]

    def __str__(self) -> str:
        return f"""\
Good words: {', '.join(self.good_words)}
Bad words: {', '.join(self.bad_words)}"""


class Clue(BaseModel):
    one_word_clue: str
    num_words: int


class SFTSample(BaseModel):
    game: Game
    clue: Clue


class ParseError(BaseModel):
    response: str


class EvaluationError(BaseModel):
    reason: str


class Evaluation(BaseModel):
    game: Game
    clue: Clue | ParseError
    reward: float
    guesses: list[str] | EvaluationError


def generate_game() -> Game:
    words = random.sample(WORDS, 20)
    good_words, bad_words = (words[:10], words[10:])
    return Game(
        good_words=good_words,
        bad_words=bad_words,
    )
