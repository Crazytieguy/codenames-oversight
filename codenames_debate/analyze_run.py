from pathlib import Path

import pandas as pd
import typer

from .models import PreferenceSet

app = typer.Typer()


@app.command()
def main(dataset_folder: Path):
    data = [
        oversight.model_dump() | {"phase": int(file.stem[-1])}
        for file in dataset_folder.glob("eval-preference-sets-*.jsonl")
        for line in file.read_text().splitlines()
        for oversight in PreferenceSet.model_validate_json(line).oversights
    ]
    df = pd.DataFrame(data)
    df["overseer_mistake"] = df["expected_score"] > df["ground_truth_score"]
    print(
        df.groupby("phase").aggregate(
            mean_ground_truth_score=("ground_truth_score", "mean"),
            mean_expected_score=("expected_score", "mean"),
            comparisons_performed=("comparisons_performed", "sum"),
            overseer_mistakes=("overseer_mistake", "mean"),
        )
        # .to_csv()
    )


if __name__ == "__main__":
    app()
