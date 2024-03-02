from codenames_debate.models import Clue, Evaluation, Game
from codenames_debate.oversight import robust_overseer


def test_robust_oversight():
    game = Game(
        good_words=["a", "b", "c", "d", "e"],
        bad_words=["f", "g", "h", "i", "j"],
    )
    evaluation = Evaluation(
        game=game,
        clue=Clue(clue="clue", targets=["a", "b", "c"]),
        score=3,
        guesses=["a", "b", "c", "f"],
    )
    oversight = robust_overseer(evaluation)
    assert oversight.expected_score == 3
    assert oversight.comparisons_performed == 7
    evaluation = Evaluation(
        game=game,
        clue=Clue(clue="clue", targets=["a", "b", "c", "d"]),
        score=3,
        guesses=["a", "b", "c", "g"],
    )
    oversight = robust_overseer(evaluation)
    assert oversight.expected_score == 0
    assert oversight.comparisons_performed == 5
