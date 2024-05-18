from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from datasets import Dataset
from peft import AutoPeftModelForCausalLM  # type: ignore
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

from .models import PreferenceSet

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    dataset_file: str,
    model_dir: str,
    output_dir: str,
    reference_model: Optional[str] = None,
    adversarial_alpha: Annotated[
        float,
        typer.Argument(
            help="How much to reward the model for performing poorly on the ground truth."
        ),
    ] = 0.0,
):
    data = [
        dpo_row
        for p_set in map(
            PreferenceSet.model_validate_json,
            Path(dataset_file).read_text().splitlines(),
        )
        if (dpo_row := p_set.dpo_row(adversarial_alpha)) is not None
    ]
    dataset = Dataset.from_list(data)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        is_trainable=True,
        adapter_name="default",
    )
    if reference_model is None:
        reference_model = model_dir
    model.load_adapter(reference_model, adapter_name="reference", is_trainable=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # critical for memory usage
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        logging_steps=1,
        num_train_epochs=1,
        max_steps=-1,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    dpo_trainer = DPOTrainer(
        model,
        model_adapter_name="default",
        ref_adapter_name="reference",
        args=training_args,
        beta=0.1,
        train_dataset=dataset,  # type: ignore
        tokenizer=tokenizer,
        max_length=128,
        max_prompt_length=128,
    )
    dpo_trainer.train()
    dpo_trainer.save_model()


if __name__ == "__main__":
    app()
