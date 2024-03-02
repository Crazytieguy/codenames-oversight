from enum import Enum, auto

from .models import Evaluation, EvaluationError, OverSight, ParseError


class OverSeer(Enum):
    ROBUST = auto()

    def __call__(self, evaluation: Evaluation) -> OverSight:
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


def robust_overseer(evaluation: Evaluation) -> OverSight:
    comparisons_performed = 0
    if isinstance(evaluation.clue, ParseError):
        return OverSight(
            clue=evaluation.clue,
            expected_score=0,
            comparisons_performed=comparisons_performed,
        )
    worst_target = evaluation.clue.targets[0]
    for target in evaluation.clue.targets[1:]:
        order = compare_word_pair(evaluation, worst_target, target)
        comparisons_performed += 1
        if order == GuessOrder.BEFORE:
            worst_target = target
    for word in evaluation.game.bad_words:
        order = compare_word_pair(evaluation, worst_target, word)
        comparisons_performed += 1
        if order == GuessOrder.AFTER:
            return OverSight(
                clue=evaluation.clue,
                expected_score=0,
                comparisons_performed=comparisons_performed,
            )
    return OverSight(
        clue=evaluation.clue,
        expected_score=len(evaluation.clue.targets),
        comparisons_performed=comparisons_performed,
    )
