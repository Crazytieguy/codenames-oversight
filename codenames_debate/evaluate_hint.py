import json
from pathlib import Path

from openai import OpenAI

from .game import Game

openai_client = OpenAI(api_key=Path("openai-api-key.txt").read_text().strip())

SYSTEM_MESSAGE = "You are an expert CodeNames player. You always think through your guesses step by step."
FIRST_PROMPT = """\
Let's play CodeNames!

In this version of CodeNames, I will give you 6 words and a hint, and you need to classify EXACTLY 3 words as green and EXACTLY 3 as red. The hint is meant to point you to the green words and away from the red words. Note that you should guess all of the words simultaneously.

Words: {words}
Hint: {hint}

Let's verify step by step.
"""
SECOND_PROMPT = "Great! Now deliver your final answer."


deliver_final_answer_schema = {
    "type": "function",
    "function": {
        "name": "deliver_final_answer",
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


def evaluate_hint(game: Game, hint: str) -> int:
    words = game.words()

    prompt = FIRST_PROMPT.format(words=words, hint=hint)
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
        tool_choice={"type": "function", "function": {"name": "deliver_final_answer"}},
        tools=[deliver_final_answer_schema],
    )
    answer = json.loads(
        chat_completion.choices[0].message.tool_calls[0].function.arguments
    )
    correct_good_words = set(answer["green_words"]).intersection(game.good_words)
    correct_bad_words = set(answer["red_words"]).intersection(game.bad_words)
    return len(correct_good_words) + len(correct_bad_words)
