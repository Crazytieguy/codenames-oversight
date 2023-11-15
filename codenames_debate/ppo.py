import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    is_xpu_available,
    set_seed,
)

from .evaluate_clue import evaluate_clue
from .models import Game

set_seed(0)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

device_map = (
    {"": f"xpu:{Accelerator().local_process_index}"}
    if is_xpu_available()
    else {"": Accelerator().local_process_index}
)

peft_model_id = "./llama-7b-hint-giving"
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "./llama-7b-hint-giving",
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)
# model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "json", data_files="codenames_debate/ppo_dataset.jsonl", split="train"
)


def prepare_sample(sample):
    game = Game(**sample)
    sample["query"] = f"{game}\nHint:"
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


dataset = dataset.map(prepare_sample, batched=False)

# the good_words and bad_words are dropped for some reason (maybe because they're lists?),
# so this mapping will be used to retrieve them during training
game_by_query = {sample["query"]: Game(**sample) for sample in dataset}


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


ppo_config = PPOConfig(
    model_name="llama-7b-hint-giving",
    learning_rate=1e-4,
    ppo_epochs=1,
    batch_size=16,
)

ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,  # don't ignore the EOS token (see above)
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 10,
}


def hint_from_response(response: str):
    assert response.endswith(".</s>")
    return response.removesuffix(".</s>")


def simple_log(
    training_step: float,
    queries: Iterable[str],
    hints: Iterable[str],
    rewards: Iterable[float],
):
    mean_reward = numpy.mean(rewards)
    with Path("codenames_debate/ppo_simple_log.jsonl").open("a") as f:
        for query, hint, reward in zip(queries, hints, rewards):
            f.write(
                json.dumps(
                    {
                        "training_step": training_step,
                        "query": query,
                        "hint": hint,
                        "reward": reward,
                        "mean_reward": mean_reward,
                    }
                )
                + "\n"
            )


total_generate_time = 0
total_evaluate_time = 0
total_ppo_step_time = 0

for step_number, batch in tqdm(
    enumerate(ppo_trainer.dataloader), total=128 / 16, desc="Training"
):
    query_tensors = [torch.tensor(ids) for ids in batch["input_ids"]]

    #### Get response from SFTModel
    start_generate = time.time()
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, **generation_kwargs
    )
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    generate_time = time.time() - start_generate
    total_generate_time += generate_time
    print(f"{generate_time=}")

    #### Evaluate response
    start_evaluate = time.time()
    games = [game_by_query[query] for query in batch["query"]]
    hints = list(map(hint_from_response, batch["response"]))
    with ThreadPoolExecutor(max_workers=8) as ex:
        rewards = ex.map(evaluate_clue, games, hints)

    rewards = [torch.tensor(r) for r in rewards]
    evaluate_time = time.time() - start_evaluate
    total_evaluate_time += evaluate_time
    print(f"{evaluate_time=}")

    #### Run PPO step
    start_ppo_step = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    ppo_step_time = time.time() - start_ppo_step
    total_ppo_step_time += ppo_step_time
    print(f"{ppo_step_time=}")

    simple_log(step_number, batch["query"], hints, [r.item() for r in rewards])
    print(f"{total_generate_time=}")
    print(f"{total_evaluate_time=}")
    print(f"{total_ppo_step_time=}")
    print(f"Step {step_number}: mean(rewards)={numpy.mean(rewards)}")

model.save_pretrained("llama-7b-hint-giving-ppo")
