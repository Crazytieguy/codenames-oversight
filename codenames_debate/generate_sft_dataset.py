import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from openai import OpenAI
from tqdm import tqdm

from .game import Game, generate_game

openai_client = OpenAI(api_key=Path("openai-api-key.txt").read_text().strip())

PROMPT = """\
Please help me give a CodeNames hint. \
The hint should be just one word, \
and allow another player to guess as many good words \
and as few bad words as possible.

{game}"""


def gen_sample(game: Game | None = None) -> str:
    if game is None:
        game = generate_game()
    prompt = PROMPT.format(game=str(game))
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    hint = chat_completion.choices[0].message.content.strip("\"' \n").title()  # type: ignore
    return f"{game}\nHint: {hint}"


def main(
    output_file: str = "codenames_debate/sft_hint_dataset.jsonl",
    num_samples: int = 100,
    concurrency: int = 3,
):
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        tasks = [ex.submit(gen_sample) for _ in range(num_samples)]
        for task in tqdm(
            as_completed(tasks), total=num_samples, desc="Generating samples"
        ):
            sample = task.result()
            with Path(output_file).open("a") as f:
                f.write(json.dumps({"text": sample}) + "\n")


if __name__ == "__main__":
    typer.run(main)
