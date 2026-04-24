from ragen.llm_agent.ctx_manager import ContextManager
from ragen.llm_agent.es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
import json
import time
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from ragen.llm_agent.base_llm import ConcurrentLLM
from ragen.llm_agent.agent_proxy import ApiCallingWrapperWg, VllmWrapperWg, LLMAgentProxy
import ray

@hydra.main(version_base=None, config_path="../config", config_name="evaluation")
def main(config):
    # detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)

    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)

    start_time = time.time()
    rollouts = proxy.rollout(
        DataProto(
            batch=None,
            non_tensor_batch=None,
            meta_info={
                "eos_token_id": 151645,
                "pad_token_id": 151643,
                "recompute_log_prob": False,
                "do_sample": config.actor_rollout_ref.rollout.do_sample,
                "validate": True,
            },
        ),
        step_num=1,
        val=True,
    )

    end_time = time.time()

    rm_scores = rollouts.batch["rm_scores"]
    metrics = rollouts.meta_info["metrics"]
    message_list = rollouts.non_tensor_batch["messages_list"]
    avg_reward = rm_scores.sum(-1).mean().item()

    def safe_convert(obj):
        """Recursively convert any non-serializable object to JSON-safe format."""
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [safe_convert(o) for o in obj]
        if isinstance(obj, dict):
            return {k: safe_convert(v) for k, v in obj.items()}
        try:
            import torch
            import numpy as np
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        return str(obj) 

    save_dir = os.path.join(os.getcwd(), "rollout_logs")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = getattr(config.data, "name", "unknown")

    base_filename = f"Qwen3_0.6B_100_metamath"
    jsonl_path = os.path.join(save_dir, f"{base_filename}_all.jsonl")
    summary_path = os.path.join(save_dir, f"summary_{base_filename}.json")
    with open(jsonl_path, "w", encoding="utf-8") as f:

        for msg in message_list:

            json_line = json.dumps(safe_convert(msg), ensure_ascii=False)

            f.write(json_line + "\n")


    summary = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "avg_reward": round(avg_reward, 4),
        "metrics": metrics,
        "num_samples": len(message_list),
        "rollout_time_sec": round(end_time - start_time, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()