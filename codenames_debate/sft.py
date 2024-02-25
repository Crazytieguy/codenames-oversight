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

from .models import SFTSample


def format_prompt(example_raw: dict) -> dict:
    example = SFTSample.model_validate(example_raw)
    clue_word = example.clue.one_word_clue.title()
    clue_num = example.clue.num_words
    return {"text": f"{example.game}\n\nClue: {clue_word}, {clue_num}"}


def main(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str = "./models/llama-7b-clue-giving",
):
    dataset = load_dataset(
        "json", data_files="codenames_debate/sft_clue_dataset.jsonl", split="train"
    ).map(format_prompt, batched=False)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # critical for memory usage
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=1,
        num_train_epochs=1,
        max_steps=-1,
        report_to=["tensorboard"],
        save_steps=0.25,
        save_total_limit=10,
        neftune_noise_alpha=5,
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)

    # For weird reasons, this is required in order for the model to learn to output eos.
    # See https://github.com/huggingface/transformers/issues/22794
    # I don't expect to need the token '♥'
    tokenizer.add_special_tokens({"pad_token": "♥"})

    response_template = "\n\nClue:"
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
        train_dataset=dataset,  # type: ignore
        dataset_text_field="text",
        max_seq_length=256,
        peft_config=peft_config,
    )

    trainer.train()  # type: ignore
    trainer.save_model(output_dir)


if __name__ == "__main__":
    typer.run(main)
