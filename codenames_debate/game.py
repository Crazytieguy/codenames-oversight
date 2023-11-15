import random
from pathlib import Path

from pydantic import BaseModel

WORDS = Path("words.txt").read_text().splitlines()


class Game(BaseModel):
    blue_words: list[str]
    red_words: list[str]
    white_words: list[str]
    black_word: str

    def __str__(self) -> str:
        return f"""\
Blue words: {', '.join(self.blue_words)}
Red words: {', '.join(self.red_words)}
White words: {', '.join(self.white_words)}
Black word: {self.black_word}"""


def generate_game() -> Game:
    words = random.sample(WORDS, 25)
    blue_words, red_words, white_words, [black_word] = (
        words[:9],
        words[9:17],
        words[17:24],
        words[24:],
    )
    return Game(
        blue_words=blue_words,
        red_words=red_words,
        white_words=white_words,
        black_word=black_word,
    )
