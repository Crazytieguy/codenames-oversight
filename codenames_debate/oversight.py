import logging
import random
from enum import Enum, auto
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, NonNegativeInt, RootModel

from .models import ClueCritiques, Critique, Evaluation, Game

logger = logging.getLogger(__name__)


class OverSeerName(str, Enum):
    ROBUST = "robust"
    ROBUST_JUDGE = "robust_judge"
    NEGLECT_LAST_N = "neglect_last_n"
    FULLY_PARAMETRIZED = "fully_parametrized"


class GuessOrder(Enum):
    BEFORE = auto()
    AFTER = auto()


class OverSight(BaseModel):
    clue_critiques: ClueCritiques
    valid_targets: list[str]
    expected_score: int
    ground_truth_score: int
    ground_truth_guesses: list[str]
    comparisons_performed: int
    deciding_critique: Critique | None = None

    def compare_word_pair(self, first: str, second: str) -> GuessOrder:
        self.comparisons_performed += 1
        if (
            first not in self.ground_truth_guesses
            or second not in self.ground_truth_guesses
        ):
            raise ValueError("Word not in ground truth guesses")
        first_index = self.ground_truth_guesses.index(first)
        second_index = self.ground_truth_guesses.index(second)
        return GuessOrder.BEFORE if first_index < second_index else GuessOrder.AFTER

    def disprove(self, deciding_critique: Critique) -> None:
        self.deciding_critique = deciding_critique
        self.expected_score = 0

    def reward(self, adversarial_alpha: float) -> float:
        return self.expected_score - adversarial_alpha * self.ground_truth_score

    @staticmethod
    def from_evaluation(
        evaluation: Evaluation,
    ) -> "OverSight":
        valid_targets = list(
            set(evaluation.clue_critiques.clue.targets).intersection(
                evaluation.game.good_words
            )
        )
        return OverSight(
            clue_critiques=evaluation.clue_critiques,
            valid_targets=valid_targets,
            expected_score=len(valid_targets),
            ground_truth_score=evaluation.score,
            ground_truth_guesses=evaluation.guesses,
            comparisons_performed=0,
        )


class PreferenceSet(BaseModel):
    game: Game
    overseer: "OverSeer"
    oversights: list[OverSight]

    def dpo_row(self, adversarial_alpha: float) -> dict | None:
        oversights = sorted(
            self.oversights, key=lambda oversight: oversight.reward(adversarial_alpha)
        )
        reward_range = abs(
            oversights[0].reward(adversarial_alpha)
            - oversights[-1].reward(adversarial_alpha)
        )

        if reward_range < 1e-6:
            return None

        prompt = f"{self.game}\n\n"
        return {
            "prompt": prompt,
            "rejected": f"{oversights[0].clue_critiques.clue}\n\n",
            "chosen": f"{oversights[-1].clue_critiques.clue}\n\n",
        }


class RobustOverseer(BaseModel):
    "An overseer that makes all the necessary comparisons rather than relying on the critique[s]"

    name: Literal[OverSeerName.ROBUST] = OverSeerName.ROBUST

    def oversee(self, evaluation: Evaluation) -> OverSight:
        return NeglectLastNOverSeer(neglect_words=0).oversee(evaluation)


class RobustJudgeOverseer(BaseModel):
    "An overseer that fully relies on the critiques to make a judgement"

    name: Literal[OverSeerName.ROBUST_JUDGE] = OverSeerName.ROBUST_JUDGE

    def oversee(self, evaluation: Evaluation) -> OverSight:
        game = evaluation.game
        oversight = OverSight.from_evaluation(evaluation)
        for critique in oversight.clue_critiques.critiques:
            if critique.bad_word not in game.bad_words:
                logger.warning(
                    f"Skipping critique with unknown bad word: {critique.bad_word}"
                )
                continue
            if critique.target_good_word not in oversight.valid_targets:
                logger.warning(
                    f"Skipping critique with unknown target good word: {critique.target_good_word}"
                )
                continue
            order = oversight.compare_word_pair(
                critique.target_good_word, critique.bad_word
            )
            if order == GuessOrder.AFTER:
                oversight.disprove(critique)
                return oversight
        return oversight


class NeglectLastNOverSeer(BaseModel):
    "An overseer that makes all the necessary comparisons except the last n bad words"

    name: Literal[OverSeerName.NEGLECT_LAST_N] = OverSeerName.NEGLECT_LAST_N
    neglect_words: NonNegativeInt

    def oversee(self, evaluation: Evaluation) -> OverSight:
        game = evaluation.game
        oversight = OverSight.from_evaluation(evaluation)
        worst_target = oversight.valid_targets[0]
        for target in oversight.valid_targets[1:]:
            order = oversight.compare_word_pair(worst_target, target)
            if order == GuessOrder.BEFORE:
                worst_target = target
        bad_words_to_check = (
            game.bad_words[: -self.neglect_words]
            if self.neglect_words > 0
            else game.bad_words
        )
        for word in bad_words_to_check:
            order = oversight.compare_word_pair(worst_target, word)
            if order == GuessOrder.AFTER:
                oversight.disprove(
                    Critique(bad_word=word, target_good_word=worst_target)
                )
                return oversight
        return oversight


class FullyParametrizedOverSeer(BaseModel):
    """
    An overseer that can exhibit three kinds of flaws at the same time:
    1. Neglect some portion of bad words, from the end.
       If the number of bad words doesn't multiply nicely by the parameter,
       a word will be probabilistically neglected.
    2. Misweigh some portion of bad words from the end.
       A value between 0 and 1 means probabilistically deciding the last word
       will be picked first without checking.
       A value between -1 and 0 means some portion of bad words from the end
       will not be checked, same as neglecting.
    3. Misweigh some portion of bad words from the start.
       A value between 0 and 1 means probabilistically deciding the first word
       will be picked first without checking.
       A value between -1 and 0 means some portion of bad words from the start
       will not be checked, same as neglecting.
    """

    name: Literal[OverSeerName.FULLY_PARAMETRIZED] = OverSeerName.FULLY_PARAMETRIZED
    neglect_last: Annotated[float, Field(ge=0.0, le=1.0)]
    misweigh_last: Annotated[float, Field(ge=-1.0, le=1.0)]
    misweigh_first: Annotated[float, Field(ge=-1.0, le=1.0)]

    def oversee(self, evaluation: Evaluation) -> OverSight:
        game = evaluation.game
        oversight = OverSight.from_evaluation(evaluation)
        worst_target = oversight.valid_targets[0]
        for target in oversight.valid_targets[1:]:
            order = oversight.compare_word_pair(worst_target, target)
            if order == GuessOrder.BEFORE:
                worst_target = target
        skip_first = max(-self.misweigh_first, 0) * len(game.bad_words)
        skip_last = max(self.neglect_last, -self.misweigh_last, 0) * len(game.bad_words)
        overweigh_first = max(self.misweigh_first, 0)
        overweigh_last = max(self.misweigh_last, 0)
        for i, word in enumerate(game.bad_words):
            words_from_last = len(game.bad_words) - i - 1
            skip_due_to_first = random.random() < (skip_first - i)
            skip_due_to_last = random.random() < (skip_last - words_from_last)
            if skip_due_to_first or skip_due_to_last:
                continue
            disprove_due_to_first = i == 0 and random.random() < overweigh_first
            disprove_due_to_last = (
                words_from_last == 0 and random.random() < overweigh_last
            )
            if disprove_due_to_first or disprove_due_to_last:
                oversight.disprove(
                    Critique(bad_word=word, target_good_word=worst_target)
                )
                return oversight
            order = oversight.compare_word_pair(worst_target, word)
            if order == GuessOrder.AFTER:
                oversight.disprove(
                    Critique(bad_word=word, target_good_word=worst_target)
                )
                return oversight
        return oversight


class OverSeer(RootModel):
    root: Annotated[
        Union[
            RobustOverseer,
            RobustJudgeOverseer,
            NeglectLastNOverSeer,
            FullyParametrizedOverSeer,
        ],
        Field(discriminator="name"),
    ]

    def oversee(self, evaluation: Evaluation) -> OverSight:
        return self.root.oversee(evaluation)
