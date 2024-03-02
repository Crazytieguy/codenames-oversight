"""
Based mostly on https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
"""

import torch
import typer
from datasets import load_dataset
from peft import LoraConfig  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from .models import SFTClueSample

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    output_dir: str,
    dataset_file: str,
    base_model: str = "meta-llama/Llama-2-7b-hf",
):
    dataset = load_dataset("json", data_files=dataset_file, split="train").map(
        format_prompt, batched=False
    )

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # critical for memory usage
        gradient_accumulation_steps=4,
        learning_rate=1.5e-4,
        logging_steps=1,
        num_train_epochs=1,
        max_steps=-1,
        report_to=["tensorboard"],
        save_steps=0.25,
        save_total_limit=10,
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, add_eos_token=True, padding_side="right"
    )

    # For weird reasons, this is required in order for the model to learn to output eos.
    # See https://github.com/huggingface/transformers/issues/22794
    # I don't expect to need the token '♥'
    tokenizer.add_special_tokens({"pad_token": "♥"})

    response_template = "\n\nClue:"
    # skip '<s>' and '▁'
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[1:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset,  # type: ignore
        dataset_text_field="text",
        peft_config=peft_config,
    )

    trainer.train()  # type: ignore
    trainer.save_model(output_dir)


def format_prompt(sample_raw: dict) -> dict:
    sample = SFTClueSample.model_validate(sample_raw)
    text = f"""{sample.game}

{sample.clue}"""
    return {"text": text}


if __name__ == "__main__":
    app()
