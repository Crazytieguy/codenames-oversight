from concurrent.futures import ThreadPoolExecutor

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    is_xpu_available,
    set_seed,
)

from .evaluate_hint import evaluate_hint
from .game import Game

set_seed(0)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    bias="none",
    task_type="CAUSAL_LM",
)

device_map = (
    {"": f"xpu:{Accelerator().local_process_index}"}
    if is_xpu_available()
    else {"": Accelerator().local_process_index}
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)
peft_model_id = "./llama-7b-hint-giving"
model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "json", data_files="codenames_debate/ppo_dataset.jsonl", split="train"
)


def prepare_sample(sample):
    sample["game"] = Game(**sample)
    sample["query"] = f"{sample['game']}\nHint:"
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


dataset = dataset.map(prepare_sample, batched=False)

ppo_config = PPOConfig(
    model_name="llama-7b-hint-giving",
    learning_rate=1e-4,
    ppo_epochs=4,
    batch_size=32,
)

ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    train_dataset=dataset,
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


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    #### Get response from SFTModel
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute reward score
    # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # pipe_outputs = reward_model(texts)
    # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    with ThreadPoolExecutor(max_workers=8) as ex:
        rewards = ex.map(evaluate_hint, batch["game"], batch["response"])

    rewards = [torch.tensor(r) for r in rewards]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_model("llama-7b-hint-giving-ppo")
