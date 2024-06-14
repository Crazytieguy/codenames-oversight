from codenames_debate.models import Clue, ClueCritiques, Evaluation, Game
from codenames_debate.oversight import OverSeer, OverSeerName


def test_robust_oversight():
    overseer = OverSeer.model_validate({"name": OverSeerName.ROBUST})
    game = Game(
        good_words=["a", "b", "c", "d", "e"],
        bad_words=["f", "g", "h", "i", "j"],
    )
    evaluation = Evaluation(
        game=game,
        clue_critiques=ClueCritiques(clue=Clue(clue="clue", targets=["a", "b", "c"])),
        score=3,
        guesses=["a", "b", "c", "f", "d", "e", "g", "h", "i", "j"],
    )
    oversight = overseer.oversee(evaluation)
    assert oversight.expected_score == 3
    assert oversight.comparisons_performed == 7
    evaluation = Evaluation(
        game=game,
        clue_critiques=ClueCritiques(
            clue=Clue(clue="clue", targets=["a", "b", "c", "d"])
        ),
        score=3,
        guesses=["a", "b", "c", "g", "d", "e", "f", "h", "i", "j"],
    )
    oversight = overseer.oversee(evaluation)
    assert oversight.expected_score == 0
    assert oversight.comparisons_performed == 5
