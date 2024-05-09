import random
from enum import Enum, auto
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
    NEGLECT_LAST_N = "neglect_last_n"


class GuessOrder(Enum):
    BEFORE = auto()
    AFTER = auto()
    UNKNOWN = auto()


class OverSight(BaseModel):
    overseer: OverSeer
    clue_critiques: ClueCritiques
    valid_targets: list[str]
    expected_score: int
    ground_truth_score: int
    ground_truth_guesses: list[str] | EvaluationError
    comparisons_performed: int
    deciding_critique: Critique | None = None

    def compare_word_pair(self, first: str, second: str) -> GuessOrder:
        self.comparisons_performed += 1
        if isinstance(self.ground_truth_guesses, EvaluationError):
            return GuessOrder.UNKNOWN
        if first not in self.ground_truth_guesses:
            if second not in self.ground_truth_guesses:
                return GuessOrder.UNKNOWN
            return GuessOrder.AFTER
        if second not in self.ground_truth_guesses:
            return GuessOrder.BEFORE
        first_index = self.ground_truth_guesses.index(first)
        second_index = self.ground_truth_guesses.index(second)
        return GuessOrder.BEFORE if first_index < second_index else GuessOrder.AFTER

    def disprove(self, deciding_critique: Critique) -> None:
        self.deciding_critique = deciding_critique
        self.expected_score = 0

    @staticmethod
    def new(
        evaluation: Evaluation,
        overseer: OverSeer,
    ) -> "OverSight":
        valid_targets = list(
            set(evaluation.clue_critiques.clue.targets).intersection(
                evaluation.game.good_words
            )
        )
        return OverSight(
            overseer=overseer,
            clue_critiques=evaluation.clue_critiques,
            valid_targets=valid_targets,
            expected_score=0
            if isinstance(evaluation.guesses, EvaluationError)
            else len(valid_targets),
            ground_truth_score=0
            if isinstance(evaluation.guesses, EvaluationError)
            else evaluation.score,
            ground_truth_guesses=evaluation.guesses,
            comparisons_performed=0,
        )


class PreferenceSet(BaseModel):
    game: Game
    oversights: list[OverSight]

    def dpo_row(self, adversarial_alpha: float) -> dict | None:
        def reward(o: OverSight) -> float:
            return o.expected_score - adversarial_alpha * o.ground_truth_score

        oversights = sorted(self.oversights, key=reward)
        if abs(reward(oversights[0]) - reward(oversights[-1])) < 1e-6:
            return None

        prompt = f"{self.game}\n\n"
        return {
            "prompt": prompt,
            "rejected": str(oversights[0].clue_critiques.clue),
            "chosen": str(oversights[-1].clue_critiques.clue),
        }


def generate_game(num_words: int = 20) -> Game:
    words = random.sample(WORDS, num_words)
    good_words, bad_words = (words[: num_words // 2], words[num_words // 2 :])
    return Game(
        good_words=good_words,
        bad_words=bad_words,
    )
