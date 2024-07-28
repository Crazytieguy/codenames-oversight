from pathlib import Path

import pandas as pd
import typer

from .oversight import PreferenceSet

app = typer.Typer()


@app.command()
def main(dataset_file: Path):
    data = [
        oversight.model_dump()
        | {f"overseer__{k}": v for k, v in p_set.overseer.model_dump().items()}
        | {"adversarial_alpha": p_set.adversarial_alpha}
        for p_set in map(
            PreferenceSet.model_validate_json, dataset_file.read_text().splitlines()
        )
        for oversight in p_set.oversights
    ]
    df = pd.DataFrame(data)
    df["mistake"] = df["expected_score"] > df["ground_truth_score"]
    df["big_mistake"] = df["expected_score"] > df["ground_truth_score"] * 2
    group_by = [c for c in df.columns if c.startswith("overseer__")] + [
        "adversarial_alpha"
    ]
    out_df = df.groupby(group_by).aggregate(
        mean_ground_truth_score=("ground_truth_score", "mean"),
        mean_expected_score=("expected_score", "mean"),
        comparisons_performed=("comparisons_performed", "sum"),
        overseer_mistakes=("mistake", "mean"),
        overseer_big_mistakes=("big_mistake", "mean"),
    )
    print(out_df.to_csv())


if __name__ == "__main__":
    app()
