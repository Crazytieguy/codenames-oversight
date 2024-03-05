import logging
import random
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

WORDS = Path("words.txt").read_text().splitlines()


class Game(BaseModel):
    good_words: list[str]
    bad_words: list[str]

    def __str__(self) -> str:
        return f"""\
Good words: {', '.join(self.good_words)}
Bad words: {', '.join(self.bad_words)}"""


class ParseError(BaseModel):
    response: str


class Clue(BaseModel):
    clue: str
    targets: list[str]

    def __str__(self) -> str:
        return f"""\
Clue: {self.clue}
Targets: {', '.join(self.targets)}"""

    @staticmethod
    def parse_response(response: str) -> "Clue | ParseError":
        "Parse a clue from the model response"
        try:
            clue_line, targets_line = response.removesuffix("</s>").strip().split("\n")
            assert clue_line.startswith("Clue: ")
            assert targets_line.startswith("Targets: ")
            clue = clue_line[len("Clue: ") :]
            targets = targets_line[len("Targets: ") :].split(", ")
            return Clue(clue=clue, targets=targets)
        except Exception:
            logging.warning(f"Failed to parse clue: {response}")
            return ParseError(response=response)


class Critique(BaseModel):
    bad_word: str
    target_good_word: str

    def __str__(self) -> str:
        return f"Critique: {self.bad_word} > {self.target_good_word}"

    @staticmethod
    def parse_response(response: str) -> "Critique | ParseError":
        "Parse a critique from the model response"
        try:
            bad_word, target_good_word = (
                response.removesuffix("</s>")
                .removeprefix("Critique: ")
                .strip()
                .split(" > ")
            )
            return Critique(bad_word=bad_word, target_good_word=target_good_word)
        except Exception:
            logging.warning(f"Failed to parse critique: {response}")
            return ParseError(response=response)


class SFTSample(BaseModel):
    game: Game
    clue: Clue
    critique: Critique


class ClueCritiques(BaseModel):
    clue: Clue | ParseError
    critiques: list[Critique | ParseError] = Field(default_factory=list)


class InferenceSample(BaseModel):
    game: Game
    clue_critiques: list[ClueCritiques]


class EvaluationError(BaseModel):
    reason: str


class Evaluation(BaseModel):
    game: Game
    clue_critiques: ClueCritiques
    score: int
    guesses: list[str] | EvaluationError


class OverSeer(str, Enum):
    ROBUST = "robust"
    JUDGE = "judge"


class OverSight(BaseModel):
    overseer: OverSeer
    clue_critiques: ClueCritiques
    expected_score: int
    ground_truth_score: int
    ground_truth_guesses: list[str] | EvaluationError
    comparisons_performed: int
    deciding_critique: Critique | None = None

    @staticmethod
    def from_evaluation(
        evaluation: Evaluation,
        overseer: OverSeer,
        expected_score: int,
        comparisons_performed: int,
        deciding_critique: Critique | None = None,
    ) -> "OverSight":
        return OverSight(
            overseer=overseer,
            clue_critiques=evaluation.clue_critiques,
            expected_score=expected_score,
            ground_truth_score=evaluation.score,
            ground_truth_guesses=evaluation.guesses,
            comparisons_performed=comparisons_performed,
            deciding_critique=deciding_critique,
        )


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
            "rejected": format_clue_for_dpo(rejected.clue_critiques.clue),
            "chosen": format_clue_for_dpo(chosen.clue_critiques.clue),
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
