from codenames_debate.models import Clue, ClueCritiques, Evaluation, Game
from codenames_debate.oversight import (
    RobustOverSeer,
)


def test_robust_oversight():
    overseer = RobustOverSeer()
    game = Game(
        good_words=["a", "b", "c", "d", "e"],
        bad_words=["f", "g", "h", "i", "j"],
    )
    evaluation = Evaluation(
        game=game,
        clue_critiques=ClueCritiques(clue=Clue(clue="clue", targets=["a", "b", "c"])),
        score=3,
        similarities={
            "a": 0.9,
            "b": 0.8,
            "c": 0.7,
            "f": 0.6,
            "d": 0.5,
            "e": 0.4,
            "g": 0.3,
            "h": 0.2,
            "i": 0.1,
            "j": 0.0,
        },
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
        similarities={
            "a": 0.9,
            "b": 0.8,
            "c": 0.7,
            "g": 0.6,
            "d": 0.5,
            "e": 0.4,
            "f": 0.3,
            "h": 0.2,
            "i": 0.1,
            "j": 0.0,
        },
    )
    oversight = overseer.oversee(evaluation)
    assert oversight.expected_score == 0
    assert oversight.comparisons_performed == 5


def test_fully_parameterized_overseer(): ...
