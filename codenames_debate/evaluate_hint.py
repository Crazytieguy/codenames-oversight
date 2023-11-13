import json
import logging
from pathlib import Path
from threading import Lock

from openai import OpenAI

from .game import Game

openai_client = OpenAI()

SYSTEM_MESSAGE = "You are a smart and helpful English professor."
FIRST_PROMPT = """\
Let's play a word game!

I will give you 6 words and a hint. Out of the  words, three are 'green' words and three are 'red' words. The hint is meant to point you to the green words and away from the red words. You need to classify EXACTLY three words as green and EXACTLY three as red, and you will get points based on how many you get right.

Words: {words}
Hint: {hint}
"""
SECOND_PROMPT = "Great! now submit your final guess"


submit_final_guess_schema = {
    "type": "function",
    "function": {
        "name": "submit_final_guess",
        "parameters": {
            "type": "object",
            "properties": {
                "green_words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "red_words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
        },
    },
}

log_file_lock = Lock()


def evaluate_hint(game: Game, hint: str) -> float:
    words = game.words()

    prompt = FIRST_PROMPT.format(words=", ".join(words), hint=hint)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    chat_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    messages.append(chat_completion.choices[0].message)
    messages.append({"role": "user", "content": SECOND_PROMPT})

    chat_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tool_choice={"type": "function", "function": {"name": "submit_final_guess"}},
        tools=[submit_final_guess_schema],
    )

    messages.append(chat_completion.choices[0].message)
    try:
        answer = json.loads(messages[-1].tool_calls[0].function.arguments)
    except json.decoder.JSONDecodeError:
        print(f"{messages=}")
        raise
    finally:
        with log_file_lock, Path("codenames_debate/hint_evaluation_log.jsonl").open(
            "a"
        ) as f:
            serializable_messages = [
                message if isinstance(message, dict) else message.model_dump()
                for message in messages
            ]
            f.write(
                json.dumps(
                    {"game": game.model_dump(), "messages": serializable_messages}
                )
                + "\n"
            )

    if len(answer["green_words"]) != 3 or len(answer["red_words"]) != 3:
        logging.warning(
            f"Invalid hint evaluation for {game.good_words=}, {game.bad_words=}, {hint=}.\n"
            f"Got {answer['green_words']=}, {answer['red_words']=}"
        )

    correct_good_words = set(answer["green_words"]).intersection(game.good_words)
    correct_bad_words = set(answer["red_words"]).intersection(game.bad_words)
    return float(len(correct_good_words) + len(correct_bad_words))
