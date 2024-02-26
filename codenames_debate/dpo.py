from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import Dataset
from peft import AutoPeftModelForCausalLM  # type: ignore
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

from .models import Clue, EvaluationPair, ParseError


def main(
    phase: Optional[int] = None
):
    if phase is None:
        dataset_file = "data/dpo-dataset.jsonl"
        model_dir = "./models/llama-7b-clue-giving"
        output_dir = "./models/llama-7b-clue-giving-dpo"
    elif phase == 2:
        dataset_file = "data/dpo-2-dataset.jsonl"
        model_dir = "./models/llama-7b-clue-giving-dpo"
        output_dir = "./models/llama-7b-clue-giving-dpo-2"
    else:
        dataset_file = f"data/dpo-{phase}-dataset.jsonl"
        model_dir = f"./models/llama-7b-clue-giving-dpo-{phase - 1}"
        output_dir = f"./models/llama-7b-clue-giving-dpo-{phase}"
    data = Path(dataset_file).read_text().splitlines()
    dataset = Dataset.from_list([format_dpo_row(line) for line in data])
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        is_trainable=True,
        adapter_name="default",
    )
    model.load_adapter(model_dir, adapter_name="reference", is_trainable=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # critical for memory usage
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        logging_steps=1,
        num_train_epochs=1,
        max_steps=-1,
        report_to=["tensorboard"],
        save_steps=0.25,
        save_total_limit=10,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    dpo_trainer = DPOTrainer(
        model,
        model_adapter_name="default",
        ref_adapter_name="reference",
        args=training_args,
        beta=0.3,
        train_dataset=dataset,  # type: ignore
        tokenizer=tokenizer,
    )
    dpo_trainer.train()
    dpo_trainer.save_model()


def format_dpo_row(raw_sample: str) -> dict:
    evaluation_pair = EvaluationPair.model_validate_json(raw_sample)
    prompt = f"{evaluation_pair.game}\n\nClue: "
    rejected, chosen = sorted(evaluation_pair.evaluations, key=lambda e: e.reward)
    return {
        "prompt": prompt,
        "rejected": format_clue(rejected.clue),
        "chosen": format_clue(chosen.clue),
    }


def format_clue(clue: Clue | ParseError) -> str:
    if isinstance(clue, ParseError):
        return f"{clue.response}"
    return f"{clue.one_word_clue}, {clue.num_words}"


if __name__ == "__main__":
    typer.run(main)
