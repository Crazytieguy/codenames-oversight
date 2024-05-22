import logging

import backoff
import numpy as np
from openai import OpenAI, RateLimitError

from .embedding_cache import EmbeddingCache
from .models import Clue, ClueCritiques, Evaluation, EvaluationError, Game

openai_client = OpenAI()
embedding_cache = EmbeddingCache()


def evaluate_clue(game: Game, clue_critiques: ClueCritiques) -> Evaluation:
    """
    Code for evaluating CodeNames clues using OpenAI's models.
    See https://czechgames.com/files/rules/codenames-rules-en.pdf for the rules of the game.

    I've decided not to allow ending the turn prematurely, for simplicity.
    """
    clue = clue_critiques.clue
    try:
        score, guesses = evaluate_clue_inner(game, clue)
        return Evaluation(
            game=game,
            clue_critiques=clue_critiques,
            score=score,
            guesses=guesses,
        )
    except Exception as err:
        logging.warning(f"Failed to evaluate clue {err=}")
        return Evaluation(
            game=game,
            clue_critiques=clue_critiques,
            score=0,
            guesses=EvaluationError(reason=repr(err)),
        )


def evaluate_clue_inner(game: Game, clue: Clue) -> tuple[int, list[str]]:
    all_words = game.good_words + game.bad_words
    if clue.clue.upper() in all_words:
        raise ValueError(f"Clue word is in the game: {clue.clue=}")

    clue_embedding = (
        get_embedding(f"Q: {clue.clue}") - get_embedding("Q: ") + get_embedding("A: ")
    )

    guesses = sorted(
        all_words,
        key=lambda word: cosine_similarity(get_embedding(word.title()), clue_embedding),
        reverse=True,
    )
    score = 0
    for word in guesses:
        if word in game.bad_words:
            break
        score += 1

    return score, guesses


def get_embedding(word: str) -> np.ndarray:
    embedding = embedding_cache.get(word)
    if embedding is None:
        embedding = get_openai_embedding_with_backoff(word)
        embedding_cache.insert(word, embedding)
    return embedding


@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def get_openai_embedding_with_backoff(word: str) -> np.ndarray:
    response = openai_client.embeddings.create(
        input=word, model="text-embedding-3-large"
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
