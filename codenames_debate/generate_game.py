from pathlib import Path
import random
from dataclasses import dataclass

WORDS = Path("words.txt").read_text().splitlines()


@dataclass
class Game:
    good_words: list[str]
    bad_words: list[str]

    def __str__(self) -> str:
        return f"""\
Good words: {', '.join(self.good_words)}
Bad words: {', '.join(self.bad_words)}"""


def generate_game() -> Game:
    words = random.sample(WORDS, 6)
    good_words, bad_words = words[:3], words[3:]
    return Game(good_words, bad_words)
