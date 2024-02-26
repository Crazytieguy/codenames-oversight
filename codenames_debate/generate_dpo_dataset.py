from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from tqdm import tqdm

from .evaluate_clue import evaluate_clue
from .models import ClueInferenceSample, EvaluationPair


def main(
    clue_dataset: Path = Path("data/clues-for-dpo.jsonl"),
    concurrency: int = 32,
):
    "Generate a DPO dataset from a dataset of clue pairs"
    data = [
        ClueInferenceSample.model_validate_json(line)
        for line in clue_dataset.read_text().splitlines()
    ]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        pairs = [ex.submit(gen_evaluation_pair, sample) for sample in data]
        for pair in tqdm(
            as_completed(pairs), desc="Generating evaluations", total=len(pairs)
        ):
            print(pair.result().model_dump_json())


def gen_evaluation_pair(clue_inference_sample: ClueInferenceSample) -> EvaluationPair:
    evaluations = (
        evaluate_clue(clue_inference_sample.game, clue_inference_sample.clues[0]),
        evaluate_clue(clue_inference_sample.game, clue_inference_sample.clues[1]),
    )
    return EvaluationPair(
        game=clue_inference_sample.game,
        evaluations=evaluations,
    )


if __name__ == "__main__":
    typer.run(main)
