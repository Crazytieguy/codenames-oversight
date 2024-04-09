from pathlib import Path

import typer

from .models import Clue, PreferenceSet, SFTSample

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(dpo_dataset: Path):
    for line in dpo_dataset.read_text().splitlines():
        preference_pair = PreferenceSet.model_validate_json(line)
        for oversight in preference_pair.oversights:
            if oversight.deciding_critique is not None:
                assert isinstance(oversight.clue_critiques.clue, Clue)
                sft_sample = SFTSample(
                    game=preference_pair.game,
                    clue=oversight.clue_critiques.clue,
                    critique=oversight.deciding_critique,
                )
                print(sft_sample.model_dump_json())


if __name__ == "__main__":
    app()
