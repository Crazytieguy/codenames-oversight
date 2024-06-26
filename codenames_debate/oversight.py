import abc
import logging
import random
from collections.abc import Callable
from enum import Enum, auto
from typing import Annotated, Literal, Optional, Self, Union

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, model_validator
from toolz import groupby

from .evaluate_clue import (
    evaluate_clue,
)
from .models import (
    CLUE_WORDS,
    Clue,
    ClueCritiques,
    Critique,
    Evaluation,
    Game,
)

logger = logging.getLogger(__name__)


class GuessOrder(Enum):
    BEFORE = auto()
    AFTER = auto()


class OverSight(BaseModel):
    clue_critiques: ClueCritiques
    valid_targets: list[str]
    expected_score: int
    ground_truth_score: int
    ground_truth_similarities: dict[str, float]
    comparisons_performed: int
    deciding_critique: Critique | None = None

    def compare_word_pair(
        self,
        first: str,
        second: str,
        misweigh_first: float = 1.0,
        misweigh_second: float = 1.0,
    ) -> GuessOrder:
        self.comparisons_performed += 1
        first_similarity = self.ground_truth_similarities.get(first)
        second_similarity = self.ground_truth_similarities.get(second)
        if first_similarity is None:
            raise ValueError(f"{first} not in ground truth guesses")
        if second_similarity is None:
            raise ValueError(f"{second} not in ground truth guesses")
        first_similarity *= misweigh_first
        second_similarity *= misweigh_second
        return (
            GuessOrder.BEFORE
            if first_similarity > second_similarity
            else GuessOrder.AFTER
        )

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
            ground_truth_similarities=evaluation.similarities,
            comparisons_performed=0,
        )


class OverSeerBase(BaseModel, abc.ABC):
    @abc.abstractmethod
    def oversee(self, evaluation: Evaluation) -> OverSight:
        pass

    @abc.abstractmethod
    def optimal(self, game: Game, adversarial_alpha: float) -> ClueCritiques:
        pass


OverSeer = Union[
    "RobustOverSeer",
    "RobustJudgeOverSeer",
    "NeglectLastNOverSeer",
    "NegligentBiasedOverSeer",
    "NegligentBiasedJudgeOverSeer",
]


class PreferenceSet(BaseModel):
    game: Game
    overseer: Annotated[
        OverSeer,
        Field(discriminator="name"),
    ]
    oversights: list[OverSight]
    adversarial_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)

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


class RobustOverSeer(OverSeerBase):
    "An overseer that makes all the necessary comparisons rather than relying on the critique[s]"

    name: Literal["robust"] = "robust"

    def oversee(self, evaluation: Evaluation) -> OverSight:
        return NeglectLastNOverSeer(neglect_words=0).oversee(evaluation)

    def optimal(self, game: Game, adversarial_alpha: float) -> ClueCritiques:
        evals = get_all_evals(game)
        for eval in evals:
            best_bad_word_similarity = max(
                eval.similarities[word] for word in game.bad_words
            )
            optimal_targets = [
                word
                for word in game.good_words
                if eval.similarities[word] > best_bad_word_similarity
            ]
            eval.clue_critiques.clue.targets = optimal_targets

        def reward(eval: Evaluation) -> int:
            return len(eval.clue_critiques.clue.targets)

        return choose_best_clue_critique(evals, reward)


class RobustJudgeOverSeer(OverSeerBase):
    "An overseer that fully relies on the critiques to make a judgement"

    name: Literal["robust_judge"] = "robust_judge"

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

    def optimal(self, game: Game, adversarial_alpha: float) -> ClueCritiques:
        raise NotImplementedError(
            "RobustJudgeOverSeer does not have an optimal policy yet"
        )


class NeglectLastNOverSeer(OverSeerBase):
    "An overseer that makes all the necessary comparisons except the last n bad words"

    name: Literal["neglect_last_n"] = "neglect_last_n"
    neglect_words: NonNegativeInt

    def oversee(self, evaluation: Evaluation) -> OverSight:
        game = evaluation.game
        oversight = OverSight.from_evaluation(evaluation)
        if not oversight.valid_targets:
            return oversight
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

    def optimal(self, game: Game, adversarial_alpha: float) -> ClueCritiques:
        if self.neglect_words == 0:
            return RobustOverSeer().optimal(game, adversarial_alpha)
        evals = get_all_evals(game)
        for eval in evals:
            best_bad_word_similarity = max(
                eval.similarities[word]
                for word in game.bad_words[: -self.neglect_words]
            )
            optimal_targets = [
                word
                for word in game.good_words
                if eval.similarities[word] > best_bad_word_similarity
            ]
            eval.clue_critiques.clue.targets = optimal_targets

        def reward(eval: Evaluation) -> float:
            return (
                len(eval.clue_critiques.clue.targets) - adversarial_alpha * eval.score
            )

        return choose_best_clue_critique(evals, reward)


class NegligentBiasedOverSeer(OverSeerBase):
    """
    An overseer that can exhibit several kinds of flaws at the same time:
    1. neglecting to test some of the bad words
    2. misweighing (biasing) some bad words, not overlapping with the neglected words
    3. misweighing (biasing) some bad words overlapping with the neglected words (this has no effect!)

    Note that bias_neglected_words should be <= neglect_words
    and you should only input games where neglect_words + bias_non_neglected_words <= len(game.bad_words)
    """

    name: Literal["negligent_biased"] = "negligent_biased"
    neglect_words: NonNegativeInt
    bias_neglected_words: NonNegativeInt
    bias_non_neglected_words: NonNegativeInt
    bias_factor: NonNegativeFloat

    @model_validator(mode="after")
    def validate_neglect_bias_overlap(self) -> Self:
        if self.bias_neglected_words > self.neglect_words:
            raise ValueError(
                "bias_neglected_words should be <= neglect_words, "
                f"got {self.bias_neglected_words=} and {self.neglect_words=}"
            )
        return self

    def oversee(self, evaluation: Evaluation) -> OverSight:
        game = evaluation.game
        if len(game.bad_words) < self.neglect_words + self.bias_non_neglected_words:
            raise ValueError(
                "Too many words to neglect and bias"
                f"got {len(game.bad_words)=}, {self.neglect_words=}, {self.bias_non_neglected_words=}"
            )
        oversight = OverSight.from_evaluation(evaluation)
        if not oversight.valid_targets:
            return oversight
        worst_target = oversight.valid_targets[0]
        for target in oversight.valid_targets[1:]:
            order = oversight.compare_word_pair(worst_target, target)
            if order == GuessOrder.BEFORE:
                worst_target = target

        for i, word in enumerate(game.bad_words):
            words_from_last = len(game.bad_words) - i - 1
            if i < self.neglect_words:
                continue

            # The overseer never biases neglected words!
            if i < self.bias_neglected_words:
                raise ValueError("More biased neglected words than neglected words!")

            elif words_from_last < self.bias_non_neglected_words:
                misweigh_second = self.bias_factor
            else:
                misweigh_second = 1.0

            order = oversight.compare_word_pair(
                worst_target, word, misweigh_second=misweigh_second
            )
            if order == GuessOrder.AFTER:
                oversight.disprove(
                    Critique(bad_word=word, target_good_word=worst_target)
                )
                return oversight
        return oversight

    def optimal(self, game: Game, adversarial_alpha: float) -> ClueCritiques:
        if len(game.bad_words) < self.neglect_words + self.bias_non_neglected_words:
            raise ValueError(
                "Too many words to neglect and bias"
                f"got {len(game.bad_words)=}, {self.neglect_words=}, {self.bias_non_neglected_words=}"
            )
        evals = get_all_evals(game)
        for eval in evals:
            adjusted_similarities = eval.similarities.copy()
            for word in game.bad_words[: self.bias_neglected_words]:
                # This doesn't actually matter
                adjusted_similarities[word] *= self.bias_factor
            for word in game.bad_words[-self.bias_non_neglected_words :]:
                adjusted_similarities[word] *= self.bias_factor
            best_bad_word_similarity = max(
                adjusted_similarities[word]
                for word in game.bad_words[self.neglect_words :]
            )
            optimal_targets = [
                word
                for word in game.good_words
                if adjusted_similarities[word] > best_bad_word_similarity
            ]
            eval.clue_critiques.clue.targets = optimal_targets

        def reward(eval: Evaluation) -> float:
            return (
                len(eval.clue_critiques.clue.targets) - adversarial_alpha * eval.score
            )

        return choose_best_clue_critique(evals, reward)


class NegligentBiasedJudgeOverSeer(OverSeerBase):
    """
    Like the NegligentBiasedOverSeer, but utilizing model-generated critiques to make a judgement

    note that neglect_words has no effect in this case, as the overseer is not tasked with noticing issues.
    """

    name: Literal["negligent_biased_judge"] = "negligent_biased_judge"
    neglect_words: NonNegativeInt
    bias_neglected_words: NonNegativeInt
    bias_non_neglected_words: NonNegativeInt
    bias_factor: NonNegativeFloat

    @model_validator(mode="after")
    def validate_neglect_bias_overlap(self) -> Self:
        if self.bias_neglected_words > self.neglect_words:
            raise ValueError(
                "bias_neglected_words should be <= neglect_words, "
                f"got {self.bias_neglected_words=} and {self.neglect_words=}"
            )
        return self

    def oversee(self, evaluation: Evaluation) -> OverSight:
        game = evaluation.game
        if len(game.bad_words) < self.neglect_words + self.bias_non_neglected_words:
            raise ValueError(
                "Too many words to neglect and bias"
                f"got {len(game.bad_words)=}, {self.neglect_words=}, {self.bias_non_neglected_words=}"
            )
        oversight = OverSight.from_evaluation(evaluation)
        if not oversight.valid_targets:
            return oversight

        bias_map = {}
        for i, word in enumerate(game.bad_words):
            words_from_last = len(game.bad_words) - i - 1
            if i < self.bias_neglected_words:
                bias_map[word] = self.bias_factor
            elif words_from_last < self.bias_non_neglected_words:
                bias_map[word] = self.bias_factor
            else:
                bias_map[word] = 1.0

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
                critique.target_good_word,
                critique.bad_word,
                misweigh_second=bias_map[critique.bad_word],
            )
            if order == GuessOrder.AFTER:
                oversight.disprove(critique)
                return oversight
        return oversight

    def optimal(self, game: Game, adversarial_alpha: float) -> ClueCritiques:
        if len(game.bad_words) < self.neglect_words + self.bias_non_neglected_words:
            raise ValueError(
                "Too many words to neglect and bias"
                f"got {len(game.bad_words)=}, {self.neglect_words=}, {self.bias_non_neglected_words=}"
            )
        evals = get_all_evals(game)
        for eval in evals:
            adjusted_similarities = eval.similarities.copy()
            for word in game.bad_words[: self.bias_neglected_words]:
                adjusted_similarities[word] *= self.bias_factor
            for word in game.bad_words[-self.bias_non_neglected_words :]:
                adjusted_similarities[word] *= self.bias_factor
            best_bad_word = max(game.bad_words, key=adjusted_similarities.__getitem__)
            optimal_targets = [
                word
                for word in game.good_words
                if adjusted_similarities[word] > adjusted_similarities[best_bad_word]
            ]
            if not optimal_targets:
                continue
            worst_target = min(optimal_targets, key=adjusted_similarities.__getitem__)
            eval.clue_critiques.clue.targets = optimal_targets
            eval.clue_critiques.critiques = [
                Critique(bad_word=best_bad_word, target_good_word=worst_target)
            ]

        def reward(eval: Evaluation) -> float:
            return (
                len(eval.clue_critiques.clue.targets) - adversarial_alpha * eval.score
            )

        return choose_best_clue_critique(evals, reward)


def choose_best_clue_critique(
    evals: list[Evaluation], reward: Callable[[Evaluation], float | int]
):
    grouped: dict[int, list[Evaluation]] = groupby(
        lambda eval: float_bucket(reward(eval)), evals
    )
    best_score = max(grouped)
    best_evals = grouped[best_score]
    chosen_eval = random.choice(best_evals)
    return chosen_eval.clue_critiques


def get_all_evals(game: Game) -> list[Evaluation]:
    raise NotImplementedError(
        "The clue dictionary is super big now, so it's best to find a different way to do this"
    )
    return [
        evaluate_clue(game, ClueCritiques(clue=Clue(clue=word, targets=[])))
        for word in CLUE_WORDS
    ]


def float_bucket(value: float) -> int:
    return int(value * 100)
