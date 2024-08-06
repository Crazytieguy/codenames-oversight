"""
Based mostly on https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
"""

from enum import Enum

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

from .models import SFTSample

app = typer.Typer(pretty_exceptions_show_locals=False)


class ModelRole(str, Enum):
    CLUE_GIVER = "clue_giver"
    CRITIQUER = "critiquer"


@app.command()
def main(
    output_dir: str,
    dataset_file: str,
    model_role: ModelRole = ModelRole.CLUE_GIVER,
    base_model: str = "meta-llama/Llama-2-7b-hf",
):
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    format_prompt = (
        format_cluer_prompt
        if model_role == ModelRole.CLUE_GIVER
        else format_critiquer_prompt
    )
    dataset = dataset.map(format_prompt, batched=False)

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
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=1,
        max_steps=-1,
        report_to=["tensorboard"],
    )

    peft_config = LoraConfig(
        r=256,
        lora_alpha=128,
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

    response_template = (
        "\n\nClue:" if model_role == ModelRole.CLUE_GIVER else "\n\nCritique:"
    )
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
        max_seq_length=256,
    )

    trainer.train()  # type: ignore
    trainer.save_model(output_dir)


def format_cluer_prompt(sample_raw: dict) -> dict:
    sample = SFTSample.model_validate(sample_raw)
    text = f"""{sample.game}

{sample.clue}

"""
    return {"text": text}


def format_critiquer_prompt(sample_raw: dict) -> dict:
    sample = SFTSample.model_validate(sample_raw)
    text = f"""{sample.game}

{sample.clue}

{sample.critique}
"""
    return {"text": text}


if __name__ == "__main__":
    app()
