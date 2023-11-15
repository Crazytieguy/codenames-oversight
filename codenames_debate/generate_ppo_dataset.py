from pathlib import Path

import typer

from .models import generate_game


def main(
    output_file: str = "codenames_debate/ppo_dataset.jsonl",
    num_samples: int = 100,
):
    samples = [generate_game().model_dump_json() + "\n" for _ in range(num_samples)]
    with Path(output_file).open("a") as f:
        f.writelines(samples)


if __name__ == "__main__":
    typer.run(main)
