import itertools
import random

import tiktoken
from openai import OpenAI

from .models import Clue, Evaluation, Game

openai_client = OpenAI()
gpt_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

PROMPT = """\
Here is a list of words:

{words}

Which one is most closely related to the word "{clue}"? Respond with just this word.
"""


def compute_reward(game: Game, guesses: list[str]) -> float:
    # The rest are guaranteed to be blue words
    last_guess = guesses[-1]

    last_guess_is_blue = last_guess in game.blue_words
    last_guess_is_red = last_guess in game.red_words
    last_guess_is_white = last_guess in game.white_words
    last_guess_is_black = last_guess == game.black_word

    last_guess_reward = {
        last_guess_is_blue: 0,
        last_guess_is_white: -1,
        last_guess_is_red: -2,
        last_guess_is_black: -9,
    }[True]

    return len(guesses) + last_guess_reward - 1


def evaluate_clue(game: Game, clue: Clue) -> Evaluation:
    remaining_words = (
        game.blue_words + game.red_words + game.white_words + [game.black_word]
    )
    random.shuffle(remaining_words)

    guesses = []

    def is_first_guess():
        return len(guesses) == 0

    def has_guesses_remaining():
        return len(guesses) < clue.num_words

    retry_count = 0

    while has_guesses_remaining() and (
        is_first_guess() or guesses[-1] in game.blue_words
    ):
        prompt = PROMPT.format(
            clue=clue.one_word_clue,
            words=" ".join(remaining_words),
        )

        word_tokens = [gpt_tokenizer.encode(word) for word in remaining_words]
        allowed_tokens = list(set(itertools.chain.from_iterable(word_tokens)))

        chat_completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            n=1,
            # This should help guarantee that the model returns a valid guess
            logit_bias={k: 1 for k in allowed_tokens},  # type: ignore
            max_tokens=max(len(tokens) for tokens in word_tokens),
        )

        guess = chat_completion.choices[0].message.content.strip(" '\"").upper()  # type: ignore

        guess_is_valid = guess in remaining_words

        if not guess_is_valid:
            if retry_count >= 1:
                raise ValueError(
                    f"OpenAI model is returning invalid guesses: {remaining_words=}, {guess=}"
                )
            retry_count += 1
            continue

        guesses.append(guess)
        remaining_words.remove(guess)

    reward = compute_reward(game, guesses)
    return Evaluation(game=game, clue=clue, reward=reward, guesses=guesses)


if __name__ == "__main__":
    from pathlib import Path

    from .models import SFTSample

    examples = Path("codenames_debate/sft_hint_dataset.jsonl").read_text().splitlines()

    for example in examples[:10]:
        sample = SFTSample.model_validate_json(example)
        evaluation = evaluate_clue(sample.game, sample.clue)
        print(evaluation.model_dump_json())
