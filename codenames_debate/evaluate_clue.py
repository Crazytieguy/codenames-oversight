import itertools
import logging
import random

import backoff
import tiktoken
from openai import OpenAI, RateLimitError

from .models import Clue, ClueCritiques, Evaluation, EvaluationError, Game

openai_client = OpenAI()
openai_tokenizers = {
    "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "gpt-4-turbo-preview": tiktoken.encoding_for_model("gpt-4-turbo-preview"),
}

PROMPT = """\
Here is a list of words:

{words}

Which one is most closely related to the word "{clue}"? Respond with just this word.
"""


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
    remaining_words = game.good_words + game.bad_words
    if clue.clue.upper() in remaining_words:
        raise ValueError(f"Clue word is in the game: {clue.clue=}")

    random.shuffle(remaining_words)

    guesses = []

    retry_count = 0
    model = "gpt-3.5-turbo"
    # start low, as I've noticed high values can cause a timeout
    logit_bias = 2.0

    while len(guesses) == 0 or guesses[-1] in game.good_words:
        prompt = PROMPT.format(
            clue=clue.clue,
            words=" ".join(remaining_words),
        )

        tokenizer = openai_tokenizers[model]
        word_tokens = [tokenizer.encode(word) for word in remaining_words]
        allowed_tokens = list(set(itertools.chain.from_iterable(word_tokens)))

        chat_completion = chat_completion_with_backoff(
            model, logit_bias, prompt, word_tokens, allowed_tokens
        )

        guess = chat_completion.choices[0].message.content.strip(" '\"").upper()  # type: ignore

        guess_is_valid = guess in remaining_words

        if not guess_is_valid:
            for word in remaining_words:
                if guess in word.split():
                    guess = word
                    break
            else:
                if retry_count == 0:
                    logging.warn(
                        f"GPT-3.5-Turbo returned an invalid guess. Retrying with higher logit bias ({guess=})"
                    )
                    logit_bias = 4.0

                elif retry_count == 1:
                    logging.warn(
                        f"GPT-3.5-Turbo returned an invalid guess twice. Retrying with GPT-4 ({guess=})"
                    )
                    model = "gpt-4-turbo-preview"
                else:
                    raise ValueError(
                        f"OpenAI models are returning invalid guesses: {remaining_words=}, {guess=}"
                    )

                retry_count += 1
                continue

        guesses.append(guess)
        remaining_words.remove(guess)

    score = len(guesses) - 1  # the last word is a bad word, thus doesn't count
    return score, guesses


@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def chat_completion_with_backoff(
    model: str,
    logit_bias: float,
    prompt: str,
    word_tokens: list[list[int]],
    allowed_tokens: list[int],
):
    return openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # This should help the model returns a valid guess
        logit_bias={k: logit_bias for k in allowed_tokens},  # type: ignore
        temperature=0.0,
        max_tokens=max(len(tokens) for tokens in word_tokens),
        timeout=10,
    )
