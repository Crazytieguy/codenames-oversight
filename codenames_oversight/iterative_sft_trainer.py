# Based on https://github.com/huggingface/trl/blob/v0.9.6/trl/trainer/iterative_sft_trainer.py
# Modified in order to accept a peft model
import logging
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from peft import PeftModel  # type: ignore
from torch.utils.data import DataLoader
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput
from trl.core import PPODecorators
from trl.import_utils import is_peft_available

logger = logging.getLogger(__name__)


class IterativeSFTTrainer(Trainer):
    """
    The IterativeSFTTrainer can be used to finetune models with methods that requires some steps between optimization.

    Attributes:
        **model** (`PreTrainedModel`) -- Model to be optimized, either an 'AutoModelForCausalLM' or an 'AutoModelForSeq2SeqLM'.
            Check the documentation of `PreTrainedModel` for more details.
        **args** (`transformers.TrainingArguments`): -- The arguments to use for training.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **optimizers** (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`): -- The optimizer and scheduler to use for training.
        **data_collator** (Union[DataCollatorForLanguageModeling, DataCollatorForSeq2Seq], *optional*) -- Data collator to be used for training and
            passed along the dataloader.
        **eval_dataset** (`datasets.Dataset`): The dataset to use for evaluation.
        **max_length** (`int`, defaults to `None`): -- The maximum length of the input.
        **truncation_mode** (`str`, defaults to `keep_end`): -- The truncation mode to use, either `keep_end` or `keep_start`.
        **preprocess_logits_for_metrics** (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`): -- The function to use to preprocess the logits before computing the metrics.
        **compute_metrics** (`Callable[[EvalPrediction], Dict]`, *optional*): -- The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to metric values.
        **optimize_device_cache ** (`bool`, *optional*, defaults to `False`) -- Optimize CUDA cache for slightly more memory-efficient training.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, PeftModel]] = None,
        args: Optional[TrainingArguments] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        max_length: Optional[int] = None,
        truncation_mode: Optional[str] = "keep_end",
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        optimize_device_cache: Optional[bool] = False,
    ):
        # Step 0: check positional arguments validity
        if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizerBase like a PreTrainedTokenizer or a PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        if not isinstance(model, (PreTrainedModel, PeftModel)):
            raise ValueError(f"model must be a PreTrainedModel, got {type(model)}")
        if not model.can_generate():
            warnings.warn(
                f"The current model class {type(model)} is not compatible with `.generate()`"
                "Please make sure that this is intended."
            )
        if optimizers[1] is None and args.max_steps == -1:
            raise ValueError(
                "When no scheduler is provided, you need to set the total number of training steps to perform `max_steps`"
            )

        self.is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        self.tokenizer = tokenizer

        if data_collator is None:
            if self.is_encoder_decoder:
                warnings.warn(
                    "No data collator is provided. Using 'DataCollatorForSeq2Seq' with"
                    "'labels_pad_token_id' set to '-100' and 'pad_to_multiple_of' set to 8."
                )
                self.data_collator = DataCollatorForSeq2Seq(
                    tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8
                )
            else:
                warnings.warn(
                    "No data collator is provided. Using 'DataCollatorForLanguageModeling'"
                )
                self.data_collator = DataCollatorForLanguageModeling(
                    self.tokenizer, mlm=False
                )
        else:
            self.data_collator = data_collator

        self.max_length = max_length
        self.truncation_mode = truncation_mode
        self.optimize_device_cache = optimize_device_cache

        super().__init__(
            model=model,
            args=args,
            data_collator=self.data_collator,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.create_optimizer_and_scheduler(self.args.max_steps)

        # prepare model, optimizer and lr_scheduler
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        self.tokenizer.truncation_side = (
            "left" if self.truncation_mode == "keep_end" else "right"
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        PPODecorators.optimize_device_cache = self.optimize_device_cache

    def prepare_model_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        if attention_mask is None:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]

        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [
                    {"input_ids": ids, "attention_mask": att, "labels": lab}
                    for ids, att, lab in zip(input_ids, attention_mask, labels)
                ]
            ).to(self.model.device)

            input_data.pop(
                "decoder_input_ids", None
            )  # This is directly computed inside the model

            input_data["labels"][
                input_data["labels"] == self.tokenizer.pad_token_id
            ] = -100

        else:
            input_data = self.data_collator(
                [
                    {"input_ids": ids, "attention_mask": att}
                    for ids, att in zip(input_ids, attention_mask)
                ]
            ).to(self.model.device)

        # truncate in case the user has provided input_ids, attention_mask and labels
        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                input_data = {k: v[: self.max_length] for k, v in input_data.items()}
            elif self.truncation_mode == "keep_end":
                input_data = {k: v[-self.max_length :] for k, v in input_data.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        return input_data

    @PPODecorators.empty_device_cache()
    def step(self, texts: List[str]):
        """
        Run an optimisation step given a list of input_ids, attention_mask, and labels or a list of text and text_labels.
        Args:
            texts (List[`str`], *optional*):
                List of strings containing the text input (if not provided, input_ids will directly be used)
        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        self.model.train()

        if self.state.global_step == 0:
            self.tr_loss = torch.tensor(0.0).to(self.args.device)
            self._globalstep_last_logged = self.state.global_step

        model_inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids, attention_mask = (
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
        )

        labels = input_ids

        model_inputs = self.prepare_model_inputs(input_ids, attention_mask, labels)

        model_inputs_names = list(model_inputs.keys())

        batch_dict = {}
        batch_dict.update(model_inputs)

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                if key in ["input_ids", "attention_mask", "labels"]:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(
                        self.model.device
                    )
            return return_dict

        batch_data = Dataset.from_dict(batch_dict)
        batch_data.set_format("torch")

        step_dataloader = DataLoader(
            batch_data,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        for batch in step_dataloader:
            with self.accelerator.accumulate(self.model):
                inputs = {k: batch[k] for k in model_inputs_names}

                loss = self.compute_loss(self.model, inputs)

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                logger.debug(f"IterativeSFTTrainer Loss: {loss}")

                tr_loss_step = loss.detach()

                loss.backward()

                if (
                    self.accelerator.sync_gradients
                    and self.args.max_grad_norm is not None
                ):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )

                # update stats etc
                self.tr_loss += tr_loss_step

        sum_norm = 0.0
        count_require_grad = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no grad"
                assert (
                    param.grad.norm().item() != 0
                ), f"{name} has grad norm 0 before update"
                sum_norm += param.grad.norm().item()
                count_require_grad += 1

        logger.debug(
            f"IterativeSFTTrainer mean grad norm: {sum_norm / count_require_grad}"
        )

        # TODO: setting sync_gradients to True is currently needed to get the optimizer not to skip the update
        # But there's probably a better way to do this
        self.optimizer.gradient_state._set_sync_gradients(True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is None, f"{name} has grad after zero_grad"

        self.state.global_step += 1

        self._maybe_log_save_evaluate()

    def _maybe_log_save_evaluate(self):
        # check if eval is required
        if self.args.eval_steps is not None:
            if (
                self.state.global_step % self.args.eval_steps == 0
                and self.state.global_step != 0
            ):
                self.evaluate(self.eval_dataset)

        # check if logging is required
        if self.args.logging_steps is not None:
            if (
                self.state.global_step % self.args.logging_steps == 0
                and self.state.global_step != 0
            ):
                logs: Dict[str, float] = {}

                tr_loss_scalar = self._nested_gather(self.tr_loss).mean().item()

                # reset tr_loss to zero
                self.tr_loss -= self.tr_loss

                logs["loss"] = round(
                    tr_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
                logs["learning_rate"] = self._get_learning_rate()

                self._globalstep_last_logged = self.state.global_step

                self.log(logs)
