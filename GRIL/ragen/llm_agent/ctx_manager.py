from itertools import zip_longest

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import re
from verl import DataProto
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

register_resolvers()


def _get_masks_and_scores_by_cumsum(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    all_scores: List[List[float]],
    use_turn_scores: bool,
    enable_response_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    name = tokenizer.name_or_path.lower()

    if "qwen" in name:
        try:
            special_token = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
            reward_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        except Exception:
            raise ValueError(f"Tokenizer {name} 似乎是 Qwen，但无法编码 <|im_start|>")
    elif "llama" in name:
        special_token = 128006
        reward_token = 128009
    else:
        raise ValueError(f"不支持的 cumsum 模型: {name}")

    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)

    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    else:
        loss_mask = turn_indicators > 1

    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)

    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores

        if "qwen" in name:
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)

    score_tensor = score_tensor[:, 1:]
    loss_mask = loss_mask[:, :-1]
    response_mask = response_mask[:, :-1]

    return score_tensor, loss_mask, response_mask


def _get_masks_and_scores_deepseek_r1_distill(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    all_scores: List[List[float]],
    use_turn_scores: bool,
    enable_response_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    USER_TOKEN_ID = 151644
    ASSISTANT_TOKEN_ID = 151645

    try:
        reward_token = tokenizer.eos_token_id
    except AttributeError:
        reward_token = 2

    bsz, seq_len = input_ids.shape
    device = input_ids.device

    response_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

    for i in range(bsz):
        is_assistant_turn = False
        found_first_user = False

        for j in range(seq_len):
            token_id = input_ids[i, j].item()

            if not found_first_user:
                if token_id == USER_TOKEN_ID:
                    found_first_user = True
                continue

            if token_id == ASSISTANT_TOKEN_ID:
                is_assistant_turn = True
            elif token_id == USER_TOKEN_ID:
                is_assistant_turn = False

            if not enable_response_mask:
                loss_mask[i, j] = True

            if is_assistant_turn:
                response_mask[i, j] = True

    if enable_response_mask:
        loss_mask = response_mask.clone()

    loss_mask[input_ids == ASSISTANT_TOKEN_ID] = False
    loss_mask[input_ids == USER_TOKEN_ID] = False
    response_mask[input_ids == ASSISTANT_TOKEN_ID] = False
    response_mask[input_ids == USER_TOKEN_ID] = False

    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)

    scores = [sum(i) for i in all_scores]
    score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)

    score_tensor = score_tensor[:, 1:]
    loss_mask = loss_mask[:, :-1]
    response_mask = response_mask[:, :-1]

    return score_tensor, loss_mask, response_mask


def get_masks_and_scores(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    all_scores: List[List[float]] = None,
    use_turn_scores: bool = False,
    enable_response_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    name = tokenizer.name_or_path.lower()

    if "distill" in name:
        return _get_masks_and_scores_deepseek_r1_distill(
            input_ids, tokenizer, all_scores, use_turn_scores, enable_response_mask
        )
    if "qwen" in name or "llama" in name:
        return _get_masks_and_scores_by_cumsum(
            input_ids, tokenizer, all_scores, use_turn_scores, enable_response_mask
        )
    if "deepseek" in name:
        try:
            return _get_masks_and_scores_by_cumsum(
                input_ids, tokenizer, all_scores, use_turn_scores, enable_response_mask
            )
        except Exception:
            raise ValueError(
                f"DeepSeek 模型 {name} 无法使用 cumsum 逻辑，请为其添加专用的处理函数。"
            )
    raise ValueError(f"get_masks_and_scores 不支持的模型: {name}")


class ContextManager:
    def __init__(self, config, tokenizer, processor=None, mode: str = "train"):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = [
            "<think>",
            "</think>",
            "<answer>",
            "</answer>",
            "<|im_start|>",
            "<|im_end|>",
        ]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
            env_tag: n_group * self.es_cfg.group_size
            for n_group, env_tag in zip(
                self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags
            )
        }
        self._init_prefix_lookup()

    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(
                f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script."
            )

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k, v in env_config.items():
                env_config_new[k] = v

            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = (
                    "\nThe meaning of each symbol in the state is:\n"
                    + ", ".join(
                        [f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()]
                    )
                )
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = (
                    "\nYour available actions are:\n"
                    + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                )
                action_lookup_str += (
                    f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" "
                    + self.action_sep
                    + " \"\n"
                )
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {
                "max_tokens": env_config.get(
                    "max_tokens", self.config.actor_rollout_ref.rollout.response_length
                )
            }

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group

        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response(self, response: str):
        think_content = ""
        answer_content = ""
        llm_response = response

        answer_end_tag = "</answer>"
        answer_start_tag = "<answer>"
        last_answer_end_pos = response.rfind(answer_end_tag)

        if last_answer_end_pos != -1:
            last_answer_start_pos = response.rfind(answer_start_tag, 0, last_answer_end_pos)
            if last_answer_start_pos != -1:
                start_slice = last_answer_start_pos + len(answer_start_tag)
                answer_content = response[start_slice:last_answer_end_pos]
                if self.config.agent_proxy.enable_think:
                    think_content = response[:last_answer_start_pos]
        else:
            answer_content = response

        for special_token in self.special_token_list:
            answer_content = answer_content.replace(special_token, "").strip()
            if self.config.agent_proxy.enable_think:
                think_content = think_content.replace(special_token, "").strip()

        if self.config.agent_proxy.enable_think:
            llm_response = f"<think>{think_content}</think><answer>{answer_content}</answer>"
        else:
            llm_response = f"<answer>{answer_content}</answer>"

        actions = []

        boxed_match = re.findall(r"\\boxed\{(.*?)\}", answer_content, re.DOTALL)
        actions = [b.strip() for b in boxed_match if b.strip()]

        if not actions:
            numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", answer_content)
            if numbers:
                actions = [numbers[-1].strip()]
            else:
                cleaned_answer = answer_content.strip()
                if cleaned_answer:
                    actions = [cleaned_answer]

        return llm_response, actions

    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        assert (
            self.config.agent_proxy.use_turn_scores is False
        ), "Reward normalization is not supported for use_turn_scores == True"

        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")

        if method == "mean_std":
            def norm_func(x):
                std = x.std(dim=-1, keepdim=True)
                if std.abs().max() > 1e-6:
                    return (x - x.mean(dim=-1, keepdim=True)) / (std + 1e-6)
                return torch.zeros_like(x)
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            def norm_func(x):
                std = x.std(dim=-1, keepdim=True)
                z = (x - x.mean(dim=-1, keepdim=True)) / (std + 1e-6) if std.abs().max() > 1e-6 else torch.zeros_like(x)
                return z.clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        group2index = {}
        for i, env_tag in enumerate(group_tags):
            group2index.setdefault(env_tag, []).append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor(
            [env_output.get("penalty", 0) for env_output in env_outputs],
            dtype=torch.float32,
        )
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]:
            for _, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores
        return score_tensor

    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        llm_input_texts = []
        messages_list = []

        for env_output in env_outputs:
            if "state" in env_output["history"][-1] and prepare_for_update:
                env_output["history"] = env_output["history"][:-1]

            max_k = getattr(self.config.agent_proxy, "max_context_window", None)
            if isinstance(max_k, int) and max_k > 0:
                env_output["history"] = env_output["history"][-max_k:]

            messages = [
                {"role": "system", "content": "You're a helpful assistant. "},
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]},
            ]

            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    format_prompt = (
                        "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
                        if self.config.agent_proxy.enable_think
                        else "<answer> [your answer] </answer>"
                    )
                    messages[-1]["content"] += (
                        f"State:\n{content['state']}\nAlways output: {format_prompt} with no extra text. Strictly follow this format.\n"
                    )
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (
                    prepare_for_update and idx == len(env_output["history"]) - 1
                ):
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})

            assert all(msg["role"] == "assistant" for msg in messages[2::2])

            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=(not prepare_for_update), tokenize=False
            )
            llm_input_texts.append(text)
            messages_list.append(messages)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(
            llm_input_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=False,
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        if prepare_for_update:
            scores = [[i.get("reward", 0.0) for i in env_output["history"]] for env_output in env_outputs]
            score_tensor, loss_mask, response_mask = get_masks_and_scores(
                input_ids,
                self.tokenizer,
                scores,
                use_turn_scores=self.config.agent_proxy.use_turn_scores,
                enable_response_mask=self.config.enable_response_mask,
            )

            normalized_score_tensor = score_tensor
            if not self.config.agent_proxy.use_turn_scores:
                normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": input_ids[:, 1:],
            },
            batch_size=input_ids.shape[0],
        )

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask
            llm_inputs.batch["rm_scores"] = normalized_score_tensor
            llm_inputs.batch["original_rm_scores"] = score_tensor

        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    metrics.setdefault(key, []).append(value)

            mean_metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }

            for key, values in metrics.items():
                if not isinstance(values, list):
                    continue
                prefix, suffix = key.split("/", 1)
                non_zero_values = [v for v in values if v != 0]
                if non_zero_values:
                    non_zero_key = f"{prefix}/non-zero/{suffix}"
                    mean_metrics[non_zero_key] = np.mean(non_zero_values)

            mean_metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": mean_metrics}

        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and "responses" in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch["responses"], skip_special_tokens=True
            )
        else:
            responses = lm_outputs.non_tensor_batch["response_texts"]

        env_ids = lm_outputs.non_tensor_batch["env_ids"]
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append(
                {
                    "env_id": env_id,
                    "llm_raw_response": response,
                    "llm_response": llm_response,
                    "actions": actions,
                }
            )
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        return self.get_lm_inputs(env_outputs, prepare_for_update=True)


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print("ctx_manager prefix", ctx_manager.prefix_lookup)

    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5, "actions_left": 2},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0},
            ],
            "group_id": 0,
            "metrics": {},
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0},
            ],
            "group_id": 1,
            "metrics": {},
        },
    ]

    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst = ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)


if __name__ == "__main__":
    main()
