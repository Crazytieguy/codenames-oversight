import logging

from .models import Critique, Evaluation, Game, GuessOrder, OverSeer, OverSight

logger = logging.getLogger(__name__)


def oversee(
    overseer: OverSeer, evaluation: Evaluation, neglect_words: int | None = None
) -> OverSight:
    oversight = OverSight.new(evaluation, overseer)
    if oversight.expected_score == 0:
        return oversight
    if evaluation.guesses == []:
        raise ValueError("Evaluation has no guesses")
    if overseer == OverSeer.ROBUST:
        robust_overseer(evaluation.game, oversight)
    elif overseer == OverSeer.JUDGE:
        judge_overseer(evaluation.game, oversight)
    elif overseer == OverSeer.NEGLECT_LAST_N:
        if neglect_words is None:
            raise ValueError("Must specify number of words to neglect")
        neglect_last_n_overseer(evaluation.game, oversight, neglect_words)
    else:
        raise NotImplementedError(f"Unknown overseer: {overseer}")
    return oversight


def robust_overseer(game: Game, oversight: OverSight):
    "An overseer that makes all the necessary comparisons rather than relying on the critique[s]"
    neglect_last_n_overseer(game, oversight, 0)


def neglect_last_n_overseer(game: Game, oversight: OverSight, neglect_words: int):
    "An overseer that makes all the necessary comparisons except the last n bad words"
    worst_target = oversight.valid_targets[0]
    for target in oversight.valid_targets[1:]:
        order = oversight.compare_word_pair(worst_target, target)
        if order == GuessOrder.BEFORE:
            worst_target = target
    bad_words_to_check = (
        game.bad_words[:-neglect_words] if neglect_words > 0 else game.bad_words
    )
    for word in bad_words_to_check:
        order = oversight.compare_word_pair(worst_target, word)
        if order == GuessOrder.AFTER:
            oversight.disprove(Critique(bad_word=word, target_good_word=worst_target))
            return


def judge_overseer(game: Game, oversight: OverSight):
    "An overseer that fully relies on the critiques to make a judgement"
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
            return
