"""
Based mostly on https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
"""

import torch
import typer
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, is_xpu_available

tqdm.pandas()


def main(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str = "llama-7b-hint-giving",
):
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
        base_model,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=1,
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

    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)

    # For weird reasons, this is required in order for the model to learn to output eos.
    # See https://github.com/huggingface/transformers/issues/22794
    # I don't expect to need the token '~'
    tokenizer.add_special_tokens({"pad_token": "~"})

    response_template = "\nHint:"
    # skip '<s>' and '▁'
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    typer.run(main)
