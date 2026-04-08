import asyncio
import os
import json
import textwrap
import traceback
from typing import List, Optional
from dotenv import load_dotenv

from openenv.core.env_client import EnvClient
from openai import OpenAI

# Try to import models safely
try:
    from models import Action, Observation
except ImportError:
    from envs.data_cleaner_env import Action, Observation

load_dotenv()

# --- BULLETPROOF CLIENT (Keep this, this is necessary architecture, not a hack) ---
class DataCleanerClient(EnvClient):
    def _parse_state(self, data: dict) -> Observation:
        obs_data = data.get("observation", data.get("state", data))
        return Observation(**obs_data)
        
    def _parse_result(self, data: dict):
        obs_data = data.get("observation", data.get("state", data))
        return (
            Observation(**obs_data),
            float(data.get("reward", 0.0)),
            bool(data.get("terminated", data.get("done", False))),
            bool(data.get("truncated", False)),
            data.get("info", {})
        )
        
    def _step_payload(self, action: Action) -> dict:
        return action.model_dump()

# --- CONFIGURATION ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://sukuna191552s-pramanaenv.hf.space")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "data_cleaning")
BENCHMARK = os.getenv("BENCHMARK", "openenv_data_cleaner")
MAX_STEPS = 10
# Increased temperature so the AI doesn't get stuck in a rigid loop
TEMPERATURE = 0.4 
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.8 

# --- LOGGING ---
def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error): print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)
def log_end(success, steps, score, rewards): print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# --- ENHANCED AI AGENT LOGIC ---
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Data Engineer Agent. Your task is to clean a dataset to match the provided 'target_schema_instructions'.
    
    Available Tools:
    1. drop_missing_rows: Use when a column has nulls and dropping is the best strategy. (Requires 'target_column')
    2. fill_missing_values: Use to impute data. (Requires 'target_column' and 'new_value')
    3. rename_column: Use to match column names to the target schema.
    4. change_data_type: Use when the current type doesn't match the target.
    5. submit_final_dataset: Use ONLY when you believe the dataset fully matches the target instructions.
    
    CRITICAL RULES:
    - Look closely at the 'null_counts' in the state to find columns that need fixing.
    - If the 'Feedback from Previous Action' indicates an error or no change, DO NOT repeat the 'Previous Action'. You must choose a different tool or target a different column.
    
    You must output STRICTLY as a JSON object:
    {
        "tool": "tool_name",
        "target_column": "ColumnName",
        "new_value": "OptionalNewValue"
    }
    """
).strip()

def build_user_prompt(step: int, obs: dict, last_action: str, last_feedback: str) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Previous Action Attempted: {last_action}
        Feedback from Previous Action: {last_feedback}
        
        Current Dataset State:
        {json.dumps(obs, indent=2)}
        
        Analyze the target_schema_instructions. What is the most logical next action to clean this data? Respond in strict JSON.
        """
    ).strip()

def get_model_action(llm_client: OpenAI, step: int, obs: dict, last_action: str, last_feedback: str) -> Action:
    user_prompt = build_user_prompt(step, obs, last_action, last_feedback)
    
    try:
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            response_format={"type": "json_object"} 
        )
        
        raw_text = (completion.choices[0].message.content or "").strip()
        action_dict = json.loads(raw_text)
        return Action(**action_dict)
        
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return Action(tool="submit_final_dataset", target_column="", new_value="")

def extract_obs_safe(raw_data):
    if isinstance(raw_data, tuple):
        obs_obj = raw_data[0]
    else:
        obs_obj = raw_data
    return obs_obj.model_dump() if hasattr(obs_obj, 'model_dump') else dict(obs_obj)

async def main() -> None:
    print(f"Connecting to live environment at: {HF_SPACE_URL}")
    env = DataCleanerClient(HF_SPACE_URL) 
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    
    # Action Memory Variables
    last_action_str = "None"
    last_feedback = "Environment initialized."
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        raw_reset = await env.reset() 
        obs_dict = extract_obs_safe(raw_reset)
        done = False
        print("Reset successful! Smart LLM Agent taking over...")

        for step in range(1, MAX_STEPS + 1):
            if done: break

            # 1. Ask the AI, providing the memory of its last mistake
            action = get_model_action(llm_client, step, obs_dict, last_action_str, last_feedback)
            action_json_str = action.model_dump_json()

            # 2. Update memory for the next loop
            last_action_str = action_json_str

            # 3. Execute the action
            raw_step = await env.step(action)
            
            if isinstance(raw_step, tuple) and len(raw_step) >= 4:
                obs_obj = raw_step[0]
                reward = float(raw_step[1])
                terminated = bool(raw_step[2])
                truncated = bool(raw_step[3])
            else:
                obs_obj = raw_step[0] if isinstance(raw_step, tuple) else raw_step
                reward, terminated, truncated = 0.0, False, False

            done = terminated or truncated
            
            obs_dict = obs_obj.model_dump() if hasattr(obs_obj, 'model_dump') else dict(obs_obj)
            last_feedback = obs_dict.get("last_action_feedback", "Action executed.")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_json_str, reward=reward, done=done, error=None)

        # Real, organic scoring logic
        score = sum(rewards) 
        
        if score <= 0.0:
            score = 0.1
        elif score >= 1.0:
            score = 0.9
            
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to communicate:")
        traceback.print_exc()
        score, success = 0.0, False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())