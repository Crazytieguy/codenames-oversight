"""
Based mostly on https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
"""

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, is_xpu_available

tqdm.pandas()

dataset = load_dataset(
    "json", data_files="codenames_debate/sft_hint_dataset.jsonl", split="train"
)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
device_map = (
    {"": f"xpu:{Accelerator().local_process_index}"}
    if is_xpu_available()
    else {"": Accelerator().local_process_index}
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)

training_args = TrainingArguments(
    output_dir="llama-7b-hint-giving",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1.41e-5,
    logging_steps=1,
    num_train_epochs=3,
    max_steps=-1,
    report_to="none",
    save_steps=100,
    save_total_limit=10,
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=40,  # this might need to be increased
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
)

trainer.train()

trainer.save_model("output")
