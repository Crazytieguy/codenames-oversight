import random
from pathlib import Path

from pydantic import BaseModel, Field

CLUE_WORDS = Path("clue-words.txt").read_text().splitlines()
GAME_WORDS = Path("game-words.txt").read_text().splitlines()


class Game(BaseModel):
    good_words: list[str]
    bad_words: list[str]

    def __str__(self) -> str:
        return f"""\
Good words: {', '.join(self.good_words)}
Bad words: {', '.join(self.bad_words)}"""


class Clue(BaseModel):
    clue: str
    targets: list[str]

    def __str__(self) -> str:
        return f"""\
Clue: {self.clue}
Targets: {', '.join(self.targets)}"""

    @staticmethod
    def parse_response(response: str) -> "Clue":
        "Parse a clue from the model response"
        clue_line, targets_line = response.removesuffix("</s>").strip().split("\n")
        if not clue_line.startswith("Clue: "):
            raise ValueError(f"Expected 'Clue: ', got {clue_line}")
        if not targets_line.startswith("Targets: "):
            raise ValueError(f"Expected 'Targets: ', got {targets_line}")
        clue = clue_line[len("Clue: ") :]
        targets = targets_line.removeprefix("Targets: ").split(", ")
        return Clue(clue=clue, targets=targets)


class Critique(BaseModel):
    bad_word: str
    target_good_word: str

    def __str__(self) -> str:
        return f"Critique: {self.bad_word} > {self.target_good_word}"

    @staticmethod
    def parse_response(response: str) -> "Critique":
        "Parse a critique from the model response"
        bad_word, target_good_word = (
            response.removesuffix("</s>")
            .removeprefix("Critique: ")
            .strip()
            .split(" > ")
        )
        return Critique(bad_word=bad_word, target_good_word=target_good_word)


class SFTSample(BaseModel):
    game: Game
    clue: Clue
    critique: Critique


class ClueCritiques(BaseModel):
    clue: Clue
    critiques: list[Critique] = Field(default_factory=list)


class InferenceSample(BaseModel):
    game: Game
    clue_critiques: list[ClueCritiques]


class Evaluation(BaseModel):
    game: Game
    clue_critiques: ClueCritiques
    score: int
    similarities: dict[str, float]


def generate_game(num_words: int = 20) -> Game:
    words = random.sample(GAME_WORDS, num_words)
    good_words, bad_words = random.sample(
        [words[: num_words // 2], words[num_words // 2 :]], 2
    )
    return Game(
        good_words=good_words,
        bad_words=bad_words,
    )
