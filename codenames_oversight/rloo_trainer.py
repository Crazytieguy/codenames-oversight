# Based on https://github.com/huggingface/trl/blob/v0.9.6/trl/trainer/rloo_trainer.py
# The only modification here is using a reward function rather than a reward model.

import gc
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast
from datasets import Dataset
from peft import PeftModel  # type: ignore
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate,
    truncate_response,
)

logger = logging.getLogger(__name__)

INVALID_LOGPROB = 1.0


@dataclass
class TrainingStats:
    approxkl: torch.Tensor
    pg_clipfrac: torch.Tensor
    pg_loss: torch.Tensor
    vf_loss: torch.Tensor
    vf_clipfrac: torch.Tensor
    entropy: torch.Tensor
    ratio: torch.Tensor


@dataclass
class RolloutData:
    context_length: int
    query_responses: torch.Tensor
    responses: torch.Tensor
    postprocessed_responses: torch.Tensor
    padding_mask: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    sequence_lengths: torch.Tensor
    postprocessed_query_responses: torch.Tensor


@dataclass
class TrainingState:
    global_step: int
    start_time: float
    stats: TrainingStats
    rollout_data: Optional[RolloutData] = None


class RLOOTrainer(Trainer):
    generation_config: GenerationConfig
    training_state: TrainingState

    def __init__(
        self,
        config: RLOOConfig,
        tokenizer: PreTrainedTokenizer,
        model: PeftModel,
        policy_adapter: str,
        ref_policy_adapter: str,
        train_dataset: Dataset | None = None,
        train_dataset_effective_len: int | None = None,
    ) -> None:
        model.set_adapter(policy_adapter)

        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.model = model

        self.model.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.model.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.policy_adapter = policy_adapter
        self.ref_policy_adapter = ref_policy_adapter

        if train_dataset is not None:
            if train_dataset_effective_len is not None:
                raise ValueError("Only pass one of tain_dataset and train_dataset_effective_len")
            self.train_dataset = train_dataset
            self.train_dataset_len = len(train_dataset)
        else:
            if train_dataset_effective_len is None:
                raise ValueError("One of tain_dataset and train_dataset_effective_len must be passed")
            self.train_dataset = None
            self.train_dataset_len = train_dataset_effective_len
        self.data_collator = None
        self.eval_dataset = None
        self.optimizer, self.lr_scheduler = None, None
        self.callbacks = None

        #########
        # calculate various batch sizes
        #########
        # allow the users to define episodes in terms of epochs.
        if args.total_episodes is None:
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len * args.rloo_k)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size,
            args.num_mini_batches,
            "`batch_size` must be a multiple of `num_mini_batches`",
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`",
        )
        if args.whiten_rewards:
            whitening_msg = f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            assert args.local_mini_batch_size >= 8, whitening_msg
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_updates = args.total_episodes // args.batch_size
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        # avoid different timestamps across processes
        time_int = broadcast(time_tensor, 0).item()  # type: ignore
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)
        # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size,
            args.rloo_k,
            "`local_batch_size` must be a multiple of rloo_k",
        )

        #########
        # setup model, optimizer, and others
        #########
        disable_dropout_in_model(model)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        DEFAULT_CALLBACKS = [DefaultFlowCallback]
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        if self.callbacks is None:
            self.callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks,
            self.model,
            self.tokenizer,
            self.optimizer,
            self.lr_scheduler,
        )
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        if self.train_dataset is not None:
            self.dataloader = DataLoader(
                self.train_dataset,  # type: ignore
                batch_size=self.local_dataloader_batch_size,
                shuffle=True,
                collate_fn=DataCollatorWithPadding(tokenizer),
                drop_last=True,  # needed; otherwise the last batch will be of ragged shape
            )
            # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
            # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
            torch.manual_seed(args.seed)
            self.model, self.optimizer, self.dataloader = accelerator.prepare(
                self.model, self.optimizer, self.dataloader
            )
        else:
            torch.manual_seed(args.seed)
            self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
        else:
            pass
            # got ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
            # Please use the model as it is, since the model has already been set to the correct devices
            # and casted to the correct `dtype`.
            # self.ref_policy = self.ref_policy.to(self.accelerator.device)
            # self.reward_model = self.reward_model.to(self.accelerator.device)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        stats_shape = (
            args.num_ppo_epochs,
            args.num_mini_batches,
            args.gradient_accumulation_steps,
        )
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        self.training_state = TrainingState(
            global_step=0,
            start_time=time.time(),
            stats=TrainingStats(
                approxkl=torch.zeros(stats_shape, device=self.accelerator.device),
                pg_clipfrac=torch.zeros(stats_shape, device=self.accelerator.device),
                pg_loss=torch.zeros(stats_shape, device=self.accelerator.device),
                vf_loss=torch.zeros(stats_shape, device=self.accelerator.device),
                vf_clipfrac=torch.zeros(stats_shape, device=self.accelerator.device),
                entropy=torch.zeros(stats_shape, device=self.accelerator.device),
                ratio=torch.zeros(stats_shape, device=self.accelerator.device),
            ),
        )

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def rollout(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Generate responses to the queries using the policy and return the postprocessed_query_responses.
        Save everything necessary for the RL step in training_state.rollout_data.
        """
        queries = queries.to(self.accelerator.device)
        assert self.training_state.rollout_data is None

        args = self.args
        model = self.model
        tokenizer = self.tokenizer

        assert args.batch_size is not None
        assert self.lr_scheduler is not None
        assert tokenizer.pad_token_id is not None

        self.training_state.global_step += 1 * args.batch_size
        self.lr_scheduler.step()
        with torch.no_grad():
            queries = queries.repeat(args.rloo_k, 1)
            context_length = queries.shape[1]
            query_responses = []
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            sequence_lengths = []

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    model.set_adapter(self.policy_adapter)
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response, logits = generate(
                        unwrapped_model,
                        query,
                        tokenizer.pad_token_id,
                        self.generation_config,  # type: ignore
                    )
                    response = query_response[:, context_length:]

                    # use the logits during generation directly, instead of using the following
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del logits, all_logprob
                    torch.cuda.empty_cache()

                    model.set_adapter(self.ref_policy_adapter)
                    ref_output = forward(model, query_response, tokenizer.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(args.stop_token_id, tokenizer.pad_token_id, response)

                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1

                    query_responses.append(query_response)
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)

            query_responses = torch.cat(query_responses, 0)
            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

            self.training_state.rollout_data = RolloutData(
                context_length=context_length,
                padding_mask=padding_mask,
                query_responses=query_responses,
                responses=responses,
                postprocessed_responses=postprocessed_responses,
                logprobs=logprobs,
                ref_logprobs=ref_logprobs,
                sequence_lengths=sequence_lengths,
                postprocessed_query_responses=postprocessed_query_responses,
            )

            return postprocessed_query_responses

    def rl_step(self, scores: torch.Tensor, metrics: dict | None = None) -> None:
        """
        Perform the RL step using the scores.
        """
        scores = scores.to(self.accelerator.device)
        rollout_data = self.training_state.rollout_data
        assert rollout_data is not None

        model = self.model
        args = self.args
        tokenizer = self.tokenizer
        training_state = self.training_state
        stats = training_state.stats

        assert args.local_batch_size is not None
        assert args.local_mini_batch_size is not None
        assert args.per_device_train_batch_size is not None
        assert tokenizer.pad_token_id is not None
        assert self.optimizer is not None
        assert self.lr_scheduler is not None

        with torch.no_grad():
            # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_eos_token = torch.any(rollout_data.postprocessed_responses == tokenizer.eos_token_id, dim=-1)
            if args.non_eos_penalty:
                scores = torch.where(
                    contain_eos_token,
                    scores,
                    torch.full_like(scores, args.penalty_reward_value),
                )

            # 4. compute rewards
            kl = rollout_data.logprobs - rollout_data.ref_logprobs
            non_score_reward = (-args.kl_coef * kl).sum(1)
            rlhf_reward = scores + non_score_reward

            # vectorized RLOO advantages implementation
            rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
            baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
            advantages = rlhf_reward - baseline
            advantages = advantages.flatten()
            torch.cuda.empty_cache()

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        model.set_adapter(self.policy_adapter)
        for ppo_epoch_idx in range(args.num_ppo_epochs):
            b_inds = np.random.permutation(args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                    with self.accelerator.accumulate(model):
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = rollout_data.responses[micro_batch_inds]
                        mb_query_responses = rollout_data.query_responses[micro_batch_inds]
                        mb_logprobs = rollout_data.logprobs[micro_batch_inds]
                        output = forward(model, mb_query_responses, tokenizer.pad_token_id)
                        logits = output.logits[:, rollout_data.context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(
                            new_logprobs,
                            rollout_data.padding_mask[micro_batch_inds],
                            INVALID_LOGPROB,
                        )
                        new_ratio = (new_logprobs - mb_logprobs).exp()
                        new_logprobs = new_logprobs.sum(1)
                        mb_logprobs = mb_logprobs.sum(1)
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = pg_loss_max.mean()
                        loss = pg_loss
                        logger.debug(f"Loss: {loss.item()}")
                        self.accelerator.backward(loss)
                        with torch.no_grad():
                            pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            stats.approxkl[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            stats.pg_clipfrac[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            stats.pg_loss[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            stats.entropy[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            stats.ratio[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                    gradient_accumulation_idx += 1

                sum_norm = 0.0
                count_require_grad = 0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        assert param.grad is not None, f"{name} has no grad"
                        assert param.grad.norm().item() != 0, f"{name} has grad norm 0 before update"
                        sum_norm += param.grad.norm().item()
                        count_require_grad += 1

                logger.debug(f"RLOOTrainer mean grad norm: {sum_norm / count_require_grad}")

                # TODO: setting sync_gradients to True is currently needed to get the optimizer not to skip the update
                # But there's probably a better way to do this
                self.optimizer.gradient_state._set_sync_gradients(True)
                self.optimizer.step()
                self.optimizer.zero_grad()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        assert param.grad is None, f"{name} has grad after zero_grad"

                minibatch_idx += 1
                # del everything and empty cache
                # fmt: off
                del (
                    output, logits, new_all_logprobs, new_logprobs,  # type: ignore
                    logprobs_diff, ratio, pg_losses, pg_losses2,  # type: ignore
                    pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,  # type: ignore
                    mb_advantage, mb_responses, mb_query_responses, mb_logprobs,  # type: ignore
                )
                # fmt: on
                torch.cuda.empty_cache()

        with torch.no_grad():
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-rollout_data.logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.mean()
            eps = int(training_state.global_step / (time.time() - training_state.start_time))
            if metrics is None:
                metrics = {}
            metrics["eps"] = eps
            metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()  # type: ignore
            metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()  # type: ignore
            metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()  # type: ignore
            metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()  # type: ignore
            metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()  # type: ignore
            metrics["policy/approxkl_avg"] = self.accelerator.gather(stats.approxkl).mean().item()  # type: ignore
            metrics["policy/clipfrac_avg"] = self.accelerator.gather(stats.pg_clipfrac).mean().item()  # type: ignore
            metrics["loss/policy_avg"] = self.accelerator.gather(stats.pg_loss).mean().item()  # type: ignore
            metrics["loss/value_avg"] = self.accelerator.gather(stats.vf_loss).mean().item()  # type: ignore
            metrics["val/clipfrac_avg"] = self.accelerator.gather(stats.vf_clipfrac).mean().item()  # type: ignore
            metrics["policy/entropy_avg"] = self.accelerator.gather(stats.entropy).mean().item()  # type: ignore
            metrics["val/ratio"] = self.accelerator.gather(stats.ratio).mean().item()  # type: ignore
            metrics["val/ratio_var"] = self.accelerator.gather(stats.ratio).var().item()  # type: ignore
            metrics["val/num_eos_tokens"] = (rollout_data.responses == tokenizer.eos_token_id).sum().item()
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["episode"] = training_state.global_step
            self.state.epoch = training_state.global_step / self.train_dataset_len  # used by self.log
            self.state.global_step += 1
            self.log(metrics)
        del kl, mean_kl, mean_entropy, scores
        torch.cuda.empty_cache()
        gc.collect()

        training_state.rollout_data = None

    def end_train(self):
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
