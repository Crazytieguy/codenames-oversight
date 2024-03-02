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
    clue: str
    targets: list[str]

    def __str__(self) -> str:
        return f"""\
Clue: {self.clue}
Targets: {', '.join(self.targets)}"""


class SFTClueSample(BaseModel):
    game: Game
    clue: Clue


class ParseError(BaseModel):
    response: str


class EvaluationError(BaseModel):
    reason: str


class Evaluation(BaseModel):
    game: Game
    clue: Clue | ParseError
    score: int
    guesses: list[str] | EvaluationError


class ClueInferenceSample(BaseModel):
    game: Game
    clues: list[Clue | ParseError]


class OverSight(BaseModel):
    clue: Clue | ParseError
    expected_score: int
    comparisons_performed: int


class PreferencePair(BaseModel):
    game: Game
    oversights: tuple[OverSight, OverSight]

    def dpo_row(self) -> dict | None:
        if self.oversights[0].expected_score == self.oversights[1].expected_score:
            return None
        prompt = f"{self.game}\n\nClue: "
        rejected, chosen = sorted(self.oversights, key=lambda e: e.expected_score)
        return {
            "prompt": prompt,
            "rejected": format_clue_for_dpo(rejected.clue),
            "chosen": format_clue_for_dpo(chosen.clue),
        }


def format_clue_for_dpo(clue: Clue | ParseError) -> str:
    if isinstance(clue, ParseError):
        return str(clue.response)
    return str(clue).removesuffix("Clue: ")


def generate_game(num_words: int = 20) -> Game:
    words = random.sample(WORDS, num_words)
    good_words, bad_words = (words[: num_words // 2], words[num_words // 2 :])
    return Game(
        good_words=good_words,
        bad_words=bad_words,
    )
