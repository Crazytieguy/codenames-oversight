import json
import random

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)

from .models import Clue, Evaluation, Game

openai_client = OpenAI()

SYSTEM_MESSAGE = "You are an expert CodeNames player and you love to help others."
PROMPT = """\
I've just started a game of CodeNames and received the first clue: "{clue_word}", {clue_num}.
This means "{clue_word}" is related to {clue_num} words on the board.
Here is the board:
{board}

Please help me choose a word that's related to "{clue_word}".\
"""

FOLLOW_UP_PROMPT = """\
That's correct! Please help me choose another word, \
or end my turn if there are no remaining appropriate choices.
"""

INVALID_GUESS_MESSAGE = """\
{guess} is an invalid choice, please choose a different word. \
Make sure you choose a word from the board.\
"""

END_TURN_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "end_turn",
        "description": "Ends the current turn",
        "parameters": {"type": "object", "properties": {}},
    },
}


CHOOSE_WORD_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "choose_word",
        "description": "Chooses a word from the board",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {"type": "string"},
            },
            "required": ["word"],
        },
    },
}


def compute_reward(game: Game, guesses: list[str]) -> float:
    # The rest are guaranteed to be blue words
    last_guess = guesses[-1]

    last_guess_is_blue = last_guess in game.blue_words
    last_guess_is_red = last_guess in game.red_words
    last_guess_is_white = last_guess in game.white_words
    last_guess_is_black = last_guess == game.black_word

    last_guess_reward = {
        last_guess_is_blue: 0,
        last_guess_is_white: 0,
        last_guess_is_red: -1,
        last_guess_is_black: -9,
    }[True]

    return len(guesses) + last_guess_reward - 1


def format_board(words: list[str]) -> str:
    words_2d = [words[i : i + 5] for i in range(0, 25, 5)]
    return "\n".join(" ".join(row) for row in words_2d)


def evaluate_clue(game: Game, clue: Clue) -> Evaluation:
    remaining_words = (
        game.blue_words + game.red_words + game.white_words + [game.black_word]
    )
    random.shuffle(remaining_words)

    prompt = PROMPT.format(
        clue_word=clue.one_word_clue,
        clue_num=clue.num_words,
        board=format_board(remaining_words),
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    guesses = []

    def is_first_guess():
        return len(guesses) == 0

    def has_guesses_remaining():
        return len(guesses) <= clue.num_words + 1

    retry_count = 0

    while has_guesses_remaining() and (
        is_first_guess() or guesses[-1] in game.blue_words
    ):
        tools = [CHOOSE_WORD_SCHEMA]
        if not is_first_guess():
            tools.append(END_TURN_SCHEMA)

        tool_choice: ChatCompletionToolChoiceOptionParam = (
            {"type": "function", "function": {"name": "choose_word"}}
            if is_first_guess()
            else "auto"
        )
        chat_completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tool_choice=tool_choice,
            tools=tools,
        )

        message = chat_completion.choices[0].message
        messages.append(message)  # type: ignore

        if message.tool_calls is None:
            raise ValueError(f"Bad response from OpenAI API: {chat_completion=}")

        function = message.tool_calls[0].function
        if function.name == "end_turn":
            break

        try:
            guess = json.loads(function.arguments)["word"].upper()
        except (json.JSONDecodeError, KeyError):
            raise ValueError(
                f"""OpenAI response doesn't match the schema: messages=\
                    {json.dumps(messages, indent=2, default=ChatCompletionMessage.model_dump)}"""
            )

        guess_is_valid = guess in remaining_words

        if not guess_is_valid:
            if retry_count >= 1:
                raise ValueError(
                    f"""OpenAI model is returning invalid guesses: messages=\
                    {json.dumps(messages, indent=2, default=ChatCompletionMessage.model_dump)}"""
                )
            retry_count += 1
            tool_message = {
                "role": "tool",
                "content": INVALID_GUESS_MESSAGE.format(guess=guess),
                "tool_call_id": message.tool_calls[0].id,
            }
            messages.append(tool_message)
            continue

        tool_message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": "CORRECT",  # won't be sent if the guess is wrong
            "tool_call_id": message.tool_calls[0].id,
        }

        follow_up_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": FOLLOW_UP_PROMPT.format(clue_word=clue.one_word_clue),
        }

        messages.extend(
            [
                tool_message,
                follow_up_message,
            ]
        )

        guesses.append(guess)
        remaining_words.remove(guess)

    pprint(messages)
    reward = compute_reward(game, guesses)
    return Evaluation(game=game, clue=clue, reward=reward, guesses=guesses)


if __name__ == "__main__":
    from pprint import pprint

    example_game = Game(
        blue_words=[
            "PUPIL",
            "ROBIN",
            "NOVEL",
            "RING",
            "CRANE",
            "DAY",
            "UNDERTAKER",
            "BAND",
            "LITTER",
        ],
        red_words=[
            "MOUTH",
            "MARCH",
            "ANGEL",
            "ANTARCTICA",
            "PALM",
            "SERVER",
            "PISTOL",
            "OLYMPUS",
        ],
        white_words=[
            "TAIL",
            "FISH",
            "LEPRECHAUN",
            "DUCK",
            "TABLET",
            "KID",
            "KNIGHT",
        ],
        black_word="DANCE",
    )
    example_clue = Clue(one_word_clue="Story", num_words=2)
    evaluation = evaluate_clue(example_game, example_clue)
    pprint(evaluation)
