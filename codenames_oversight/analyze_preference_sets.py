from pathlib import Path

import pandas as pd
import typer

from .oversight import Evaluation, PreferenceSet

app = typer.Typer()


@app.command()
def main(dataset_file: Path):
    data = [
        oversight.model_dump()
        | {f"overseer__{k}": v for k, v in p_set.overseer.model_dump().items()}
        | {"adversarial_alpha": p_set.adversarial_alpha}
        | {
            "evaluation": Evaluation(
                game=p_set.game,
                clue_critiques=oversight.clue_critiques,
                score=oversight.ground_truth_score,
                similarities=oversight.ground_truth_similarities,
            )
        }
        for p_set in map(PreferenceSet.model_validate_json, dataset_file.read_text().splitlines())
        for oversight in p_set.oversights
    ]
    df = pd.DataFrame(data)
    df["overestimate"] = df["expected_score"] > df["ground_truth_score"]
    df["big_overestimate"] = df["expected_score"] > df["ground_truth_score"] * 2
    df["mistake"] = df.apply(is_mistake, axis=1)
    group_by = [c for c in df.columns if c.startswith("overseer__")] + ["adversarial_alpha"]
    out_df = df.groupby(group_by).aggregate(
        mean_ground_truth_score=("ground_truth_score", "mean"),
        mean_expected_score=("expected_score", "mean"),
        comparisons_performed=("comparisons_performed", "sum"),
        overseer_overestimates=("overestimate", "mean"),
        overseer_big_overestimates=("big_overestimate", "mean"),
        overseer_mistakes=("mistake", "mean"),
    )
    print(out_df.to_csv(), end="")


def is_mistake(row: pd.Series) -> bool:
    true_picks = row["evaluation"].good_picks()
    targets = row["valid_targets"]
    if len(set(targets) - set(true_picks)) > 0:
        return row["deciding_critique"] is None
    return row["deciding_critique"] is not None


if __name__ == "__main__":
    app()
