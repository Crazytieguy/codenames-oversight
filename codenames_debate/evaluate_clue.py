from typing import TYPE_CHECKING, TypeVar

import backoff
import numpy as np
from openai import OpenAI, RateLimitError

from .embedding_cache import EmbeddingCache
from .models import CLUE_WORDS, Clue, ClueCritiques, Evaluation, Game

if TYPE_CHECKING:
    _T = TypeVar("_T")

    def cache(wrapped: _T) -> _T: ...

else:
    from functools import cache

openai_client = OpenAI()
embedding_cache = EmbeddingCache()


def evaluate_clue(game: Game, clue_critiques: ClueCritiques) -> Evaluation:
    """
    Code for evaluating CodeNames clues using OpenAI's models.
    See https://czechgames.com/files/rules/codenames-rules-en.pdf for the rules of the game.

    I've decided not to allow ending the turn prematurely, for simplicity.
    """
    clue = clue_critiques.clue
    score, similarities = evaluate_clue_inner(game, clue)
    return Evaluation(
        game=game,
        clue_critiques=clue_critiques,
        score=score,
        similarities=similarities,
    )


def evaluate_clue_inner(game: Game, clue: Clue) -> tuple[int, dict[str, float]]:
    all_words = game.good_words + game.bad_words
    if clue.clue.upper() in all_words:
        raise ValueError(f"Clue word is in the game: {clue.clue=}")
    if clue.clue not in CLUE_WORDS:
        raise ValueError(f"Clue word not allowed: {clue.clue=}")

    similarities = {word: clue_word_similarity(clue.clue, word) for word in all_words}
    best_bad_word_similarity = max(similarities[word] for word in game.bad_words)
    score = sum(
        similarities[word] > best_bad_word_similarity for word in game.good_words
    )

    return score, similarities


@cache
def clue_word_similarity(clue: str, word: str) -> float:
    return cosine_similarity(get_clue_embedding(clue), get_game_word_embedding(word))


@cache
def get_game_word_embedding(word: str) -> np.ndarray:
    return get_embedding(word.title())


def get_clue_embedding(clue: str) -> np.ndarray:
    return get_embedding(f"Q: {clue}") - Q_COLON_EMBEDDING + A_COLON_EMBEDDING


def get_embedding(word: str) -> np.ndarray:
    embedding = embedding_cache.get(word)
    if embedding is None:
        embedding = get_openai_embedding_with_backoff(word)
        embedding_cache.insert(word, embedding)
    return embedding


Q_COLON_EMBEDDING = get_embedding("Q: ")
A_COLON_EMBEDDING = get_embedding("A: ")


@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def get_openai_embedding_with_backoff(word: str) -> np.ndarray:
    response = openai_client.embeddings.create(
        input=word, model="text-embedding-3-large"
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
