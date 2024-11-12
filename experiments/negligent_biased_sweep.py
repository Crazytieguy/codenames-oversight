import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from codenames_oversight.evaluate_clue import evaluate_clue
from codenames_oversight.models import Game
from codenames_oversight.oversight import (
    CLUE_WORDS_INDEXABLE,
    NegligentBiasedBaseOverSeer,
    NegligentBiasedJudgeOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
)

logging.basicConfig(level=logging.INFO)


def main(vocab_file: Optional[str] = None):
    games = [Game.model_validate_json(line) for line in sys.stdin]
    all_overseers = [
        overseer_class(
            neglect_words=neglect_words,
            bias_neglected_words=bias_neglected_words,
            bias_non_neglected_words=bias_non_neglected_words,
            bias_factor=bias_factor,
            neglect_good_words=neglect_good_words,
        )
        for overseer_class in [
            NegligentBiasedJudgeOverSeer,
            NegligentBiasedOverSeer,
            NegligentBiasedBaseOverSeer,
        ]
        for neglect_words, bias_neglected_words, bias_non_neglected_words, bias_factor, neglect_good_words in [
            (0, 0, 0, 1.0, 0),
            (1, 0, 0, 1.0, 0),
            (2, 0, 0, 1.0, 0),
            (0, 0, 2, 0.5, 0),
            (0, 0, 1, 1.2, 0),
            (0, 0, 0, 1.0, 2),
            (1, 1, 0, 2.0, 0),
        ]
    ]
    if vocab_file is not None:
        # Dict from clue word to count
        vocab: dict[str, int] = json.loads(Path(vocab_file).read_text())
        all_clue_words = [w for w, count in vocab.items() for _ in range(count)]
    else:
        all_clue_words = CLUE_WORDS_INDEXABLE
    for num_clue_words in tqdm(
        [
            # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1024,
            2048,
        ],
        desc="Running negligent biased sweep",
    ):
        for overseer in tqdm(all_overseers, desc=f"{num_clue_words} clue words"):
            for p_set in run_params(games, overseer, all_clue_words, num_clue_words):
                print(p_set.model_dump_json())


def run_params(games: list[Game], overseer: OverSeer, all_clue_words: list[str], num_clue_words: int):
    psets = []
    for game in tqdm(games):
        retries = 10
        while True:
            try:
                clue_words = random.sample(all_clue_words, num_clue_words)
                psets.extend(gen_optimal_preference_sets(overseer, game, clue_words))
                break
            except ValueError:
                retries -= 1
                if retries <= 0:
                    raise
    return psets


def gen_optimal_preference_sets(
    overseer: OverSeer,
    game: Game,
    clue_words: list[str],
) -> list[PreferenceSet]:
    clue_critiques_by_alpha = overseer.optimal(game, clue_words, [i * 0.05 for i in range(20)])
    oversights = {
        adversarial_alpha: overseer.oversee(evaluate_clue(game, clue_critiques))
        for adversarial_alpha, clue_critiques in clue_critiques_by_alpha.items()
    }
    return [
        PreferenceSet(
            game=game,
            overseer=overseer,
            oversights=[oversight],
            adversarial_alpha=adversarial_alpha,
            optimization_strength=len(clue_words),
        )
        for adversarial_alpha, oversight in oversights.items()
    ]


if __name__ == "__main__":
    typer.run(main)
