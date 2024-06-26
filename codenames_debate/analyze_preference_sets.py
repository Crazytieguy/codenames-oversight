from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .oversight import PreferenceSet

app = typer.Typer()


@app.command()
def main(dataset_file: Path, group_by: Optional[list[str]] = None):
    data = [
        oversight.model_dump()
        | p_set.overseer.model_dump()
        | {"adversarial_alpha": p_set.adversarial_alpha}
        for p_set in map(
            PreferenceSet.model_validate_json, dataset_file.read_text().splitlines()
        )
        for oversight in p_set.oversights
    ]
    df = pd.DataFrame(data)
    df["overseer_mistake"] = df["expected_score"] > df["ground_truth_score"]
    df["overseer_big_mistake"] = df["expected_score"] > df["ground_truth_score"] * 2
    if group_by:
        out_df = df.groupby(group_by).aggregate(
            mean_ground_truth_score=("ground_truth_score", "mean"),
            mean_expected_score=("expected_score", "mean"),
            comparisons_performed=("comparisons_performed", "sum"),
            overseer_mistakes=("overseer_mistake", "mean"),
            overseer_big_mistakes=("overseer_big_mistake", "mean"),
        )
        print(out_df.to_csv())
    else:
        out_s: pd.Series = df.aggregate(
            mean_ground_truth_score=("ground_truth_score", "mean"),
            mean_expected_score=("expected_score", "mean"),
            comparisons_performed=("comparisons_performed", "sum"),
            overseer_mistakes=("overseer_mistake", "mean"),
            overseer_big_mistakes=("overseer_big_mistake", "mean"),
        )  # type: ignore
        print(out_s.to_csv())


if __name__ == "__main__":
    app()
