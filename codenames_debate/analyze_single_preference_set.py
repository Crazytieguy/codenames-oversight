from pathlib import Path

import pandas as pd
import typer

from .oversight import PreferenceSet

app = typer.Typer()


@app.command()
def main(dataset_file: Path):
    data = [
        oversight.model_dump()
        for line in dataset_file.read_text().splitlines()
        for oversight in PreferenceSet.model_validate_json(line).oversights
    ]
    df = pd.DataFrame(data)
    df["overseer_mistake"] = df["expected_score"] > df["ground_truth_score"]
    mean_ground_truth_score = df["ground_truth_score"].mean()
    mean_expected_score = df["expected_score"].mean()
    comparisons_performed = df["comparisons_performed"].sum()
    overseer_mistakes = df["overseer_mistake"].mean()
    print(f"""\
mean_ground_truth_score: {mean_ground_truth_score:0.2f}
mean_expected_score:     {mean_expected_score:0.2f}
overseer_mistakes:       {overseer_mistakes:0.2f}
comparisons_performed:   {comparisons_performed}
""")


if __name__ == "__main__":
    app()
