import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from tqdm import tqdm

from .models import SFTSample, generate_game

openai_client = OpenAI()

SYSTEM_MESSAGE = "You are an expert CodeNames player."
PROMPT = """\
Let's play CodeNames!

{game}

Please give a clue for the blue team. \
The clue is a single word, followed by a number. \
The number is the number of blue words that the clue applies to. \
Make sure your clue is not associated with any of the red words."""

SUBMIT_CLUE_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "submit_clue",
        "parameters": {
            "type": "object",
            "properties": {
                "one_word_clue": {"type": "string"},
                "num_words": {"type": "integer"},
            },
        },
    },
}


def main(
    output_file: str = "codenames_debate/sft_clue_dataset.jsonl",
    num_samples: int = 100,
    concurrency: int = 8,
):
    "Generate the supervised fine-tuning dataset for clue giving."
    with Path(output_file).open("a") as f, ThreadPoolExecutor(
        max_workers=concurrency
    ) as ex:
        tasks = [ex.submit(gen_sample) for _ in range(num_samples)]
        for task in tqdm(
            as_completed(tasks), total=num_samples, desc="Generating samples"
        ):
            sample = task.result()
            f.write(sample.model_dump_json() + "\n")


def gen_sample() -> SFTSample:
    game = generate_game()
    prompt = PROMPT.format(game=str(game).replace("Good", "Blue").replace("Bad", "Red"))
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        tool_choice={"type": "function", "function": {"name": "submit_clue"}},
        tools=[SUBMIT_CLUE_SCHEMA],
    )
    clue = chat_completion.choices[0].message.tool_calls[0].function.arguments  # type: ignore

    return SFTSample(game=game, clue=json.loads(clue))


if __name__ == "__main__":
    typer.run(main)
