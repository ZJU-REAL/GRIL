import re
import random
import gym
import zlib
from collections import defaultdict
from datasets import load_from_disk,load_dataset
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed

class PremiseDetectEnv(BaseLanguageBasedEnv):
    def __init__(self,
                 config,
                 max_steps_premise: int = 4,
                 min_reasoning_length: int = 50,
                 empty_reasoning_penalty: float = -1.0,
                 short_reasoning_penalty: float = -0.5,
                 repetition_threshold: float = 0.4,
                 repetition_penalty: float = -1.0,
                 inconsistency_penalty: float = -0.5,
                 max_steps_full: int = 4,
                 diversity_penalty_lambda: float = 0.5,
                 hallucination_penalty: float = -1.0):
        
        super().__init__()
        self.config = config
        self.max_steps_premise = max_steps_premise
        self.min_reasoning_length = min_reasoning_length
        self.empty_reasoning_penalty = empty_reasoning_penalty
        self.short_reasoning_penalty = short_reasoning_penalty
        self.repetition_threshold = repetition_threshold
        self.repetition_penalty = repetition_penalty
        self.inconsistency_penalty = inconsistency_penalty
        self.max_steps_full = max_steps_full
        self.diversity_penalty_lambda = diversity_penalty_lambda
        self.hallucination_penalty = hallucination_penalty

        self.dataset = load_from_disk(self.config.dataset_path)
        self._initialize_episode_states()

    def _initialize_episode_states(self):
        self.mode = None
        self.current_question = None
        self.correct_answer = None
        self.render_cache = ""
        self.missed_premise = None
        self.stage = 1
        self.stage_1_steps = 0
        self.stage_2_steps = 0
        self.premise_detected = False
        self._stage_1_rewards = []
        self._stage_2_rewards = []
        self.step_num = 0
        self._step_rewards = []
        self._unique_answers_count = defaultdict(int)
        self._total_valid_answers = 0
        self.missing_steps = 0

    def reset(self,step_num,val, seed=None):
        self._initialize_episode_states()
        with all_seed(seed):
            question_data = self.dataset[random.randint(0, len(self.dataset) - 1)]
        if not val:
            if question_data.get('removed_premise') is not None:
                self.mode = 'missing'
                self.current_question = question_data['question']
                self.missed_premise = question_data.get('removed_premise', "")
                self.correct_answer = self._extract_answer(question_data.get('answer'))
            else:
                self.mode = 'full'
                self.current_question = question_data['full_question']
                self.correct_answer = self._extract_answer(question_data.get('answer'))
        else:
            if question_data.get('removed_premise') is not None:
                self.mode = 'missing'
                self.current_question = question_data['question']
                self.missed_premise = question_data.get('removed_premise', "")
                self.correct_answer = self._extract_answer(question_data.get('answer'))
            else:
                self.mode = 'full'
                self.current_question = question_data['full_question']
                self.correct_answer = self._extract_answer(question_data.get('answer'))
            
        self.render_cache = self.current_question
        

        return self.render_cache

    def step(self, reasoning: str , action: str):
        if self.mode is None:
            raise RuntimeError("You must call reset() before calling step()")

        if self.mode == 'missing':
            if reasoning is None:
                raise ValueError("Reasoning must be provided for 'missing' mode.")
            return self._step_premise_detect(reasoning, action)
        elif self.mode == 'full':
            return self._step_full_problem(reasoning,action)
        else:
            raise ValueError(f"Unknown environment mode: {self.mode}")

    def _step_premise_detect(self, reasoning: str, action: str):
        if (not reasoning or not reasoning.strip()) and (not action or not action.strip()):
            obs = f"Penalty ({self.empty_reasoning_penalty}): Both reasoning and action are empty."
            return self._apply_penalty_and_get_response(self.empty_reasoning_penalty, obs)
        
        if self.stage == 1:
            return self._step_stage_1(reasoning, action)
        else:
            return self._step_stage_2(reasoning, action)

    def _step_stage_1(self, reasoning: str, action: str):
        prefix = self.mode
        self.stage_1_steps += 1
        detected = self._detects_missing_premise(reasoning, action)
        if detected:
            self.premise_detected = True
            self.missing_steps = self.stage_1_steps
            reward = 1.0 * (0.3 ** (self.stage_1_steps - 1))
            self.stage = 2
            observation = f"Successfully detected a missing premise. Here is the missed information: {self.missed_premise}. Please solve the problem now.You should give detailed reasonig steps"
        else:
            reward = 0.0
            observation = "That is incorrect. Please try again."
        self._stage_1_rewards.append(reward)
        info = {f"{prefix}/detected_premise": self.premise_detected, f"{prefix}/current_stage": self.stage}
        done = (self.stage_1_steps + self.stage_2_steps) >= self.max_steps_premise and not self.premise_detected
        if done:
            penalty = -1.0
            self._stage_1_rewards.append(penalty)
            reward = penalty
            
            if not self.premise_detected:
                self.missing_steps = self.max_steps_premise

            info[f"{prefix}/final_total_reward"] = sum(self._stage_1_rewards)
            info[f"{prefix}/success"] = False
            info[f"{prefix}/detected_premise"] = self.premise_detected
            info[f"{prefix}/missing_steps"] = self.missing_steps
        self.render_cache = observation
        return observation, reward, done, info

    def _step_stage_2(self, reasoning: str, action: str):
        prefix = self.mode
        if self._detects_missing_premise(reasoning, action):
            penalty = -2.0
            self._stage_2_rewards.append(penalty)
            total_reward = 0.3 * sum(self._stage_1_rewards) + 0.7 * sum(self._stage_2_rewards)
            observation = "Penalty: Incorrectly detected a missing premise in Stage 2."
            done = True
            info = {
                f"{prefix}/final_total_reward": total_reward, 
                f"{prefix}/success": False, 
                f"{prefix}/detected_premise": self.premise_detected,
                f"{prefix}/missing_steps": self.missing_steps
            }
            self.render_cache = observation
            return observation, penalty, done, info

        self.stage_2_steps += 1
        is_correct = self._normalize_answer(action) in self._normalize_answer(self.correct_answer)
        reward = 1.0 if is_correct else 0.0
        self._stage_2_rewards.append(reward)
        if action is not None:
            self._unique_answers_count[self._normalize_answer(action)] += 1
            self._total_valid_answers += 1

        done = is_correct or (self.stage_1_steps + self.stage_2_steps) >= self.max_steps_premise
        if done:
            T = self._total_valid_answers
            E = len(self._unique_answers_count)
            diversity_penalty = self.diversity_penalty_lambda * (1 - (E / T)) if T > 0 else 0.0
            total_reward = 0.3 * sum(self._stage_1_rewards) + 0.7 * sum(self._stage_2_rewards) - diversity_penalty
            observation = "Correct!" if is_correct else "Episode finished."
            info = {
                f"{prefix}/final_total_reward": total_reward,
                f"{prefix}/diversity_penalty": diversity_penalty,
                f"{prefix}/success": is_correct,
                f"{prefix}/detected_premise": self.premise_detected,
                f"{prefix}/missing_steps": self.missing_steps
            }
        else:
            observation = f"Incorrect. Remember, the missing info was: {self.missed_premise}. Try again."
            info = {f"{prefix}/success": is_correct, f"{prefix}/detected_premise": self.premise_detected}
        self.render_cache = observation
        return observation, reward, done, info

    def _step_full_problem(self, reasoning:str,action: str):
        prefix = self.mode
        if self._detects_missing_premise(reasoning=reasoning, action=action):
            penalty = self.hallucination_penalty
            observation = f"Penalty ({penalty}): Incorrectly detected a missing premise in a full problem. The episode has ended."
            done = True
            info = {
                f"{prefix}/success": False,
                f"{prefix}/final_total_reward": penalty,
                f"{prefix}/hallucination_penalty": penalty,
            }
            self.render_cache = observation
            return observation, penalty, done, info

        is_correct = self._normalize_answer(action) in self._normalize_answer(self.correct_answer)
        is_valid = self._normalize_answer(action) != ""
        reward = 1.0 if is_correct else 0.0
        info = {f"{prefix}/action_is_valid": is_valid, f"{prefix}/success_step": is_correct}
        
        if is_valid:
            self._unique_answers_count[self._normalize_answer(action)] += 1
            self._total_valid_answers += 1
            self._step_rewards.append(reward)
        self.step_num += 1

        done = is_correct or self.step_num >= self.max_steps_full
        if done:
            T = self._total_valid_answers
            E = len(self._unique_answers_count)
            penalty = self.diversity_penalty_lambda * (1 - (E / T)) if T > 0 else 0.0
            total_reward = sum(self._step_rewards) - penalty
            observation = "Correct!" if is_correct else "Episode finished."
            info = {
                f"{prefix}/final_total_reward": total_reward,
                f"{prefix}/diversity_penalty": penalty,
                f"{prefix}/success": is_correct
            }
            self.render_cache = observation
            return observation, total_reward, done, info
        else:
            observation = "Incorrect. Please think again."
            done = False
            self.render_cache = observation
            return observation, reward, done, info

    def _extract_answer(self, response: str):
        if response is None: return None
        patterns = [r"####\s*(.*)$", r"\\boxed\{(.*?)\}", r"The answer is: (.*?)$"]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
            if match: return match.group(1).strip()
        return None

    def _normalize_answer(self, answer: str):
        if answer is None: return ""
        return re.sub(r"\s+", "", answer.strip().lower())

    def _detects_missing_premise(self, reasoning: str, action: str) -> bool:
        reasoning_lower = reasoning.lower() if reasoning else ""
        action_lower = action.lower() if action else ""
        phrases = ["[missing premise detected]", "insufficient information", "not enough information"]
        in_reasoning = any(p in reasoning_lower for p in phrases)
        in_action = any(p in action_lower for p in phrases)
        return in_reasoning or in_action or self._normalize_answer(action) == "insufficientinformation"

    def render(self, mode='human'):
        return self.render_cache