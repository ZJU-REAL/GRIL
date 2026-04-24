from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import PIL.Image
import hydra
import random
import numpy as np

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers

register_resolvers()


@dataclass
class EnvStatus:
    truncated: bool = False
    terminated: bool = False
    num_actions: int = 0
    rewards: List[float] = field(default_factory=list)
    seed: Optional[int] = None


class EnvStateManager:
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        seed_cfg = getattr(self.sys_config, "seed", None)
        if seed_cfg is not None:
            self.base_seed = seed_cfg.get(mode, None)
        else:
            self.base_seed = None
        self.seed_counter = 0
        self._init_envs()
        self.rollout_cache = None

    def _init_envs(self):
        assert (
            sum(self.config.env_configs.n_groups) == self.env_groups
        ), f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert (
            len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups)
        ), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(
                done_groups * self.group_size, (done_groups + n_group) * self.group_size
            ):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](
                        **cfg_template.env_config
                    )
                env_obj = REGISTERED_ENVS[env_class](env_config)
                entry = {
                    "tag": tag,
                    "group_id": env_id // self.group_size,
                    "env_id": env_id,
                    "env": env_obj,
                    "config": env_config,
                    "status": EnvStatus(),
                    "max_actions_per_traj": max_actions_per_traj,
                }
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def reset(self, step_num, val, seed: Optional[int] = None):
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [
            {
                "env_id": entry["env_id"],
                "history": [],
                "group_id": entry["group_id"],
                "tag": entry["tag"],
                "penalty": 0,
            }
            for entry in envs
        ]

        if seed is None:
            if self.mode == "train":
                if self.base_seed is not None:
                    seed = self.base_seed + self.seed_counter
                    self.seed_counter += self.env_groups
                else:
                    seed = random.randint(0, 1000000)
            else:
                seed = 123 if self.base_seed is None else self.base_seed
        else:
            if self.mode == "train" and self.base_seed is not None:
                self.seed_counter = seed - self.base_seed + 1

        seeds = _expand_seed(seed)
        for seed, entry in zip(seeds, envs):
            entry["env"].reset(step_num, val, seed=seed)
            entry["status"] = EnvStatus(seed=seed)

        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env["env"].render())
            cache["history"] = self._update_cache_history(
                cache["history"],
                next_state=next_state,
                actions_left=env["max_actions_per_traj"],
                num_actions_info=None,
            )

        self.rollout_cache = rollout_cache
        return rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        def _execute_actions(env, reasoning, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(reasoning, action)
                acc_reward += reward
                turn_info.update(info)
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(
            status,
            history,
            cur_obs,
            max_actions_per_traj,
            executed_actions,
            acc_reward,
            turn_done,
            turn_info,
            env_input,
        ):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward)
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True
                status.truncated = not turn_info.get("success", False)
            history = self._update_cache_history(
                history,
                next_state=obs,
                actions_left=actions_left,
                num_actions_info={
                    "actions": executed_actions,
                    "reward": acc_reward,
                    "info": turn_info,
                    "llm_response": env_input["llm_response"],
                    "llm_raw_response": env_input["llm_raw_response"],
                },
            )
            return status, history

        envs = self.envs
        env_outputs = []

        for env_input in all_env_inputs:
            entry = envs[env_input["env_id"]]
            env_id, env = entry["env_id"], entry["env"]
            actions_left_before = entry["max_actions_per_traj"] - entry["status"].num_actions

            valid_actions = self._extract_map_valid_actions(entry, env_input["actions"])
            (
                acc_reward,
                turn_info,
                turn_done,
                executed_actions,
            ) = _execute_actions(
                env,
                env_input["llm_response"],
                valid_actions[:actions_left_before],
            )

            if len(valid_actions) != len(env_input["actions"]) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty

            status, history = _log_env_state(
                entry["status"],
                self.rollout_cache[env_id]["history"],
                entry["env"].render(),
                entry["max_actions_per_traj"],
                executed_actions,
                acc_reward,
                turn_done,
                turn_info,
                env_input,
            )
            entry["status"] = status

            if entry["status"].num_actions >= entry["max_actions_per_traj"] and not turn_done:
                entry["status"].truncated = True
                entry["status"].terminated = True
                turn_done = True

            self.rollout_cache[env_id]["history"] = history

            if not turn_done:
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def get_rollout_states(self):
        envs = self.envs
        rollout_cache = self.rollout_cache

        NO_AVERAGE_KEYWORDS = [
            "outcome/",
            "env_type_id",
            "type",
            "missing_steps",
            "detected_premise",
            "final_total_reward",
            "hallucination_penalty",
            "pass@",
            "metric_",
        ]

        TURN_LVL_METRICS = ["action_is_effective", "action_is_valid", "end_of_page"]

        for entry, cache in zip(envs, rollout_cache):
            status = entry["status"]
            env_metric = {
                "success": float(status.terminated and (not status.truncated)),
                "num_actions": status.num_actions,
            }
            custom_metric = {}

            for turn in cache["history"]:
                for k, v in turn.get("info", {}).items():
                    if k == "success":
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    try:
                        val = float(v)
                    except (ValueError, TypeError):
                        continue
                    custom_metric[k].append(val)

            for k, v in custom_metric.items():
                is_episodic_metric = any(kw in k for kw in NO_AVERAGE_KEYWORDS)

                if is_episodic_metric:
                    env_metric[k] = v[-1] if len(v) > 0 else 0.0
                elif "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                    steps = max(1, len(cache["history"]) - 1)
                    env_metric[k] = np.sum(v) / steps
                else:
                    env_metric[k] = np.sum(v)

            cache["history"][-1]["metrics"] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache["metrics"] = env_metric

            if entry["tag"] == "MetamathQA":
                cache["correct_answer"] = entry["env"].correct_answer

        group_success = {}
        for entry, cache in zip(envs, rollout_cache):
            key = (entry["tag"], entry["group_id"])
            success_val = cache["metrics"].get(f"{entry['tag']}/success", 0.0)
            group_success.setdefault(key, []).append(success_val)

        for (tag, gid), succ_list in group_success.items():
            pass_success = float(any(succ_list))
            for entry, cache in zip(envs, rollout_cache):
                if entry["tag"] == tag and entry["group_id"] == gid:
                    cache["metrics"][f"{tag}/pass@{self.group_size}"] = pass_success

        return rollout_cache

    def _update_cache_history(
        self,
        history: List[Dict],
        next_state,
        actions_left,
        num_actions_info: Optional[Dict] = None,
    ):
        if num_actions_info is not None:
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)

        entry = {}
        if isinstance(next_state, str):
            entry["state"] = next_state
        else:
            entry["state"] = "<images>" * len(next_state)
            entry["images"] = next_state
        entry["actions_left"] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        action_lookup = getattr(entry["env"].config, "action_lookup", None)
        if action_lookup is None:
            return actions
        rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
        actions = [action.lower() for action in actions]
        return [rev_action_lookup[action] for action in actions if action in rev_action_lookup]

    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        if isinstance(state, str):
            return state
        if isinstance(state, np.ndarray):
            state = [state]
        return [PIL.Image.fromarray(_state, mode="RGB") for _state in state]

    def render(self):
        return [entry["env"].render() for entry in self.envs]

    def close(self):
        for entry in self.envs:
            entry["env"].close()


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):
        print(f"Environment {i}:\n{render}\n")

    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"],
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"],
        },
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"],
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"],
        },
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):
        print(f"Environment {i}:\n{render}\n")

    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")

    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
