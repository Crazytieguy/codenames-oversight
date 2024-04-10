import logging
from enum import Enum, auto

from .models import (
    Critique,
    Evaluation,
    EvaluationError,
    OverSeer,
    OverSight,
    ParseError,
)

logger = logging.getLogger(__name__)


def oversee(
    overseer: OverSeer, evaluation: Evaluation, neglect_words: int | None = None
) -> OverSight:
    if overseer == OverSeer.ROBUST:
        return robust_overseer(evaluation)
    elif overseer == OverSeer.JUDGE:
        return judge_overseer(evaluation)
    elif overseer == OverSeer.NEGLECT_LAST_N:
        if neglect_words is None:
            raise ValueError("Must specify number of words to neglect")
        return neglect_last_n_overseer(evaluation, neglect_words)
    raise NotImplementedError(f"Unknown overseer: {overseer}")


class GuessOrder(Enum):
    BEFORE = auto()
    AFTER = auto()
    UNKNOWN = auto()


def compare_word_pair(evaluation: Evaluation, first: str, second: str) -> GuessOrder:
    if isinstance(evaluation.guesses, EvaluationError):
        return GuessOrder.UNKNOWN
    if first not in evaluation.guesses:
        if second not in evaluation.guesses:
            return GuessOrder.UNKNOWN
        return GuessOrder.AFTER
    if second not in evaluation.guesses:
        return GuessOrder.BEFORE
    first_index = evaluation.guesses.index(first)
    second_index = evaluation.guesses.index(second)
    return GuessOrder.BEFORE if first_index < second_index else GuessOrder.AFTER


def basic_validation(overseer: OverSeer, evaluation: Evaluation) -> OverSight | None:
    if isinstance(evaluation.guesses, EvaluationError):
        return OverSight.from_evaluation(evaluation, overseer, 0, 0)
    if evaluation.guesses == []:
        # Invalid clue
        return OverSight.from_evaluation(evaluation, overseer, -1, 0)
    assert not isinstance(evaluation.clue_critiques.clue, ParseError)
    for target in evaluation.clue_critiques.clue.targets:
        if target.upper() not in evaluation.game.good_words:
            return OverSight.from_evaluation(evaluation, overseer, -1, 0)
    return None


def robust_overseer(evaluation: Evaluation) -> OverSight:
    "An overseer that makes all the necessary comparisons rather than relying on the critique[s]"
    if (oversight := basic_validation(OverSeer.ROBUST, evaluation)) is not None:
        return oversight
    assert not isinstance(evaluation.clue_critiques.clue, ParseError)
    comparisons_performed = 0
    worst_target = evaluation.clue_critiques.clue.targets[0]
    for target in evaluation.clue_critiques.clue.targets[1:]:
        order = compare_word_pair(evaluation, worst_target, target)
        comparisons_performed += 1
        if order == GuessOrder.BEFORE:
            worst_target = target
    for word in evaluation.game.bad_words:
        order = compare_word_pair(evaluation, worst_target, word)
        comparisons_performed += 1
        if order == GuessOrder.AFTER:
            return OverSight.from_evaluation(
                evaluation,
                OverSeer.ROBUST,
                0,
                comparisons_performed,
                Critique(bad_word=word, target_good_word=worst_target),
            )
    return OverSight.from_evaluation(
        evaluation,
        OverSeer.ROBUST,
        len(evaluation.clue_critiques.clue.targets),
        comparisons_performed,
    )


def neglect_last_n_overseer(evaluation: Evaluation, neglect_words: int) -> OverSight:
    "An overseer that makes all the necessary comparisons except the last n bad words"
    if (oversight := basic_validation(OverSeer.NEGLECT_LAST_N, evaluation)) is not None:
        return oversight
    assert not isinstance(evaluation.clue_critiques.clue, ParseError)
    comparisons_performed = 0
    worst_target = evaluation.clue_critiques.clue.targets[0]
    for target in evaluation.clue_critiques.clue.targets[1:]:
        order = compare_word_pair(evaluation, worst_target, target)
        comparisons_performed += 1
        if order == GuessOrder.BEFORE:
            worst_target = target
    for word in evaluation.game.bad_words[:-neglect_words]:
        order = compare_word_pair(evaluation, worst_target, word)
        comparisons_performed += 1
        if order == GuessOrder.AFTER:
            return OverSight.from_evaluation(
                evaluation,
                OverSeer.ROBUST,
                0,
                comparisons_performed,
                Critique(bad_word=word, target_good_word=worst_target),
            )
    return OverSight.from_evaluation(
        evaluation,
        OverSeer.ROBUST,
        len(evaluation.clue_critiques.clue.targets),
        comparisons_performed,
    )


def judge_overseer(evaluation: Evaluation) -> OverSight:
    "An overseer that fully relies on the critiques to make a judgement"
    if (oversight := basic_validation(OverSeer.JUDGE, evaluation)) is not None:
        return oversight
    assert not isinstance(evaluation.clue_critiques.clue, ParseError)
    comparisons_performed = 0
    for critique in evaluation.clue_critiques.critiques:
        if isinstance(critique, ParseError):
            logger.warning(f"Skipping critique due to parse error: {critique}")
            continue
        if critique.bad_word not in evaluation.game.bad_words:
            logger.warning(
                f"Skipping critique with unknown bad word: {critique.bad_word}"
            )
            continue
        if critique.target_good_word not in evaluation.clue_critiques.clue.targets:
            logger.warning(
                f"Skipping critique with unknown target good word: {critique.target_good_word}"
            )
            continue
        order = compare_word_pair(
            evaluation, critique.target_good_word, critique.bad_word
        )
        comparisons_performed += 1
        if order == GuessOrder.AFTER:
            return OverSight.from_evaluation(
                evaluation, OverSeer.JUDGE, 0, comparisons_performed, critique
            )
    return OverSight.from_evaluation(
        evaluation,
        OverSeer.JUDGE,
        len(evaluation.clue_critiques.clue.targets),
        comparisons_performed,
    )
