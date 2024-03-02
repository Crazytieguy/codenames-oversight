import itertools
import logging
import random

import tiktoken
from openai import OpenAI

from .models import Clue, Evaluation, EvaluationError, Game, ParseError

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


def parse_clue(response: str) -> Clue | ParseError:
    "Parse a clue from the model response"
    try:
        clue_line, targets_line = response.removesuffix("</s>").strip().split("\n")
        assert clue_line.startswith("Clue: ")
        assert targets_line.startswith("Targets: ")
        clue = clue_line[len("Clue: ") :]
        targets = targets_line[len("Targets: ") :].split(", ")
        return Clue(clue=clue, targets=targets)
    except Exception:
        logging.warn(f"Failed to parse clue: {response}")
        return ParseError(response=response)


def evaluate_clue(game: Game, clue: Clue | ParseError) -> Evaluation:
    """
    Code for evaluating CodeNames clues using OpenAI's models.
    See https://czechgames.com/files/rules/codenames-rules-en.pdf for the rules of the game.

    I've decided not to allow ending the turn prematurely, for simplicity.
    """
    if isinstance(clue, ParseError):
        return Evaluation(
            game=game,
            clue=clue,
            score=-1,
            guesses=[],
        )
    try:
        return evaluate_clue_inner(game, clue)
    except Exception as err:
        logging.warning(f"Failed to evaluate clue {err=}")
        return Evaluation(
            game=game,
            clue=clue,
            score=0,
            guesses=EvaluationError(reason=repr(err)),
        )


def evaluate_clue_inner(game: Game, clue: Clue) -> Evaluation:
    remaining_words = game.good_words + game.bad_words
    if clue.clue.upper() in remaining_words:
        # Invalid clue
        return Evaluation(
            game=game,
            clue=clue,
            score=-1,
            guesses=[],
        )

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

        chat_completion = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # This should help the model returns a valid guess
            logit_bias={k: logit_bias for k in allowed_tokens},  # type: ignore
            temperature=0.0,
            max_tokens=max(len(tokens) for tokens in word_tokens),
            timeout=10,
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
    return Evaluation(game=game, clue=clue, score=score, guesses=guesses)
