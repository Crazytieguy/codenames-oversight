from pathlib import Path

import pandas as pd
import typer

from .models import PreferencePair

app = typer.Typer()


@app.command()
def main(dataset_file: Path):
    pairs = [
        PreferencePair.model_validate_json(line)
        for line in dataset_file.read_text().splitlines()
    ]
    df = pd.DataFrame(
        [
            {
                "expected_score": e.expected_score,
                "ground_truth_score": e.ground_truth_score,
            }
            for p in pairs
            for e in p.oversights
        ]
    )
    print(df.describe())

if __name__ == "__main__":
    app()
