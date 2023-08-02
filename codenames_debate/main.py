from datasets import Dataset
from trl import SFTTrainer

dataset = Dataset.from_json("sft_hint_dataset.json")

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,  # type: ignore
    dataset_text_field="text",
    max_seq_length=40,
)