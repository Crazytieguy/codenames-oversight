from enum import Enum, auto

from .models import Evaluation, EvaluationError, OverSight, ParseError


class OverSeer(str, Enum):
    ROBUST = auto()

    def oversee(self, evaluation: Evaluation) -> OverSight:
        if self == OverSeer.ROBUST:
            return robust_overseer(evaluation)
        raise NotImplementedError(f"Unknown overseer: {self}")


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


def basic_validation(evaluation: Evaluation) -> OverSight | None:
    if isinstance(evaluation.guesses, EvaluationError):
        return OverSight.from_evaluation(evaluation, 0, 0)
    if evaluation.guesses == []:
        # Invalid clue
        return OverSight.from_evaluation(evaluation, -1, 0)
    assert not isinstance(evaluation.clue_critiques.clue, ParseError)
    for target in evaluation.clue_critiques.clue.targets:
        if target.upper() not in evaluation.game.good_words:
            return OverSight.from_evaluation(evaluation, -1, 0)
    return None


def robust_overseer(evaluation: Evaluation) -> OverSight:
    "An overseer that makes all the necessary comparisons rather than relying on the critique[s]"
    comparisons_performed = 0
    if (oversight := basic_validation(evaluation)) is not None:
        return oversight
    assert not isinstance(evaluation.clue_critiques.clue, ParseError)
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
            return OverSight.from_evaluation(evaluation, 0, comparisons_performed)
    return OverSight.from_evaluation(
        evaluation, len(evaluation.clue_critiques.clue.targets), comparisons_performed
    )
