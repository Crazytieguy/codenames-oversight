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

    def words(self) -> list[str]:
        words = self.good_words + self.bad_words
        random.shuffle(words)
        return words


def generate_game() -> Game:
    words = random.sample(WORDS, 6)
    good_words, bad_words = words[:3], words[3:]
    return Game(good_words=good_words, bad_words=bad_words)
