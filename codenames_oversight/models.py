import random
from pathlib import Path

from pydantic import BaseModel, Field

GAME_WORDS = Path("game-words.txt").read_text().splitlines()
CLUE_WORDS = {word.title() for word in Path("nwl2023.txt").read_text().splitlines()}


class Game(BaseModel):
    good_words: list[str]
    bad_words: list[str]

    def __str__(self) -> str:
        return f"""\
Good words: {' '.join(self.good_words)}
Bad words: {' '.join(self.bad_words)}"""

    @staticmethod
    def parse(repr: str) -> "Game":
        good_words, bad_words = repr.strip().splitlines()
        return Game(
            good_words=good_words.removeprefix("Good words: ").split(),
            bad_words=bad_words.removeprefix("Bad words: ").split(),
        )


class Clue(BaseModel):
    clue: str
    targets: list[str]

    def __str__(self) -> str:
        return f"""\
Clue: {self.clue}
Targets: {' '.join(self.targets)}"""

    @staticmethod
    def parse_response(response: str) -> "Clue":
        "Parse a clue from the model response"
        clue_line, targets_line = response.removesuffix("</s>").strip().split("\n")
        if not clue_line.startswith("Clue: "):
            raise ValueError(f"Expected 'Clue: ', got {clue_line}")
        if not targets_line.startswith("Targets:"):
            raise ValueError(f"Expected 'Targets:', got {targets_line}")
        clue = clue_line[len("Clue: ") :]
        targets = [t for t in targets_line.removeprefix("Targets:").split() if t]
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
    critique: Critique | None = None


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

    def good_picks(self) -> list[str]:
        picks = []
        best_bad_word_similarity = max(
            self.similarities[word] for word in self.game.bad_words
        )
        for word in sorted(
            self.game.good_words, key=self.similarities.__getitem__, reverse=True
        ):
            if self.similarities[word] <= best_bad_word_similarity:
                break
            picks.append(word)
        return picks


def generate_game(n_good_words: int, n_bad_words: int) -> Game:
    words = random.sample(GAME_WORDS, n_good_words + n_bad_words)
    good_words, bad_words = words[:n_good_words], words[n_good_words:]
    return Game(
        good_words=good_words,
        bad_words=bad_words,
    )
