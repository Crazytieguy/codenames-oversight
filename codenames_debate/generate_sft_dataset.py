import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from tqdm import tqdm

from .models import Clue, SFTClueSample, generate_game

openai_client = OpenAI()
app = typer.Typer(pretty_exceptions_show_locals=False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = "You are an expert CodeNames player."
PROMPT = """\
Let's play CodeNames!

{game}

Please give a clue for the blue team. \
The clue is a single word. \
In addition, give a list of targetted blue words that are associated with the clue. \
Make sure your clue is not associated with any of the red words."""

SUBMIT_CLUE_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "submit_clue",
        "parameters": {
            "type": "object",
            "properties": {
                "clue": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


@app.command()
def main(
    num_samples: int = 512,
    concurrency: int = 32,
):
    "Generate the supervised fine-tuning dataset for clue giving."
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        tasks = [ex.submit(gen_sample) for _ in range(num_samples)]
        for task in tqdm(
            as_completed(tasks), total=num_samples, desc="Generating samples"
        ):
            sample = task.result()
            if sample is not None:
                print(sample.model_dump_json())


def gen_sample() -> SFTClueSample | None:
    game = generate_game()
    prompt = PROMPT.format(game=str(game).replace("Good", "Blue").replace("Bad", "Red"))
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        tool_choice={"type": "function", "function": {"name": "submit_clue"}},
        tools=[SUBMIT_CLUE_SCHEMA],
    )
    clue = chat_completion.choices[0].message.tool_calls[0].function.arguments  # type: ignore
    clue = Clue.model_validate_json(clue)
    clue.clue = clue.clue.title()
    if clue.clue.upper() in game.good_words + game.bad_words:
        logger.warning(f"Invalid clue: {clue.clue}")
        return None
    for i, target in enumerate(clue.targets):
        clue.targets[i] = target.upper()
        if target.upper() not in game.good_words:
            logger.warn(f"Invalid target word: {target}")
            return None

    return SFTClueSample(game=game, clue=clue)


if __name__ == "__main__":
    app()
