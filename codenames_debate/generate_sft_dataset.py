import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import typer

from tqdm import tqdm

from .generate_game import generate_game, Game

openai.api_key_path = "openai_api_key.txt"

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
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    hint = chat_completion.choices[0].message.content.strip('"').title()  # type: ignore
    return f"{game}\nHint: {hint}"


def main(output_file: str = "sft_hint_dataset.json", num_samples: int = 100):
    with ThreadPoolExecutor(max_workers=5) as ex:
        tasks = [ex.submit(gen_sample) for _ in range(num_samples)]
        samples = [
            task.result()
            for task in tqdm(
                as_completed(tasks), total=num_samples, desc="Generating samples"
            )
        ]
        Path(output_file).write_text(json.dumps({"text": samples}, indent=2))


if __name__ == "__main__":
    typer.run(main)
