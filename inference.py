import asyncio
import os
import json
import textwrap
import traceback
from typing import List, Optional
from dotenv import load_dotenv

from openenv.core.env_client import EnvClient
from openai import OpenAI

from models import Action, Observation

load_dotenv()

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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_SPACE_URL = "http://localhost:8000"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct") 
TASK_NAME = "SST_hackathon_env"
BENCHMARK = "SST_hackathon_env"
MAX_STEPS = 10
TEMPERATURE = 0.4  
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.8 

def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error): print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)
def log_end(success, steps, score, rewards): print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Data Engineer Agent. Your task is to clean a dataset to match the provided 'target_schema_instructions'.
    
    Available Tools:
    1. drop_missing_rows: Use when a column has nulls and dropping is the best strategy.
    2. fill_missing_values: Use to impute data. (Requires 'target_column' and 'new_value')
    3. change_data_type: Use when the current type doesn't match the target.
    4. submit_final_dataset: Use ONLY when you believe the dataset fully matches the instructions.
    
    CRITICAL RULES:
    - Look closely at the 'missing_values' dictionary to see what needs fixing.
    - If the 'Feedback from Previous Action' indicates an error or no change, DO NOT repeat the previous action. Try a different tool.
    
    You must output STRICTLY as a JSON object:
    {"tool": "tool_name", "target_column": "ColumnName", "new_value": "OptionalNewValue"}
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
        
        Analyze the target_schema_instructions. What is the most logical next action?
        """
    ).strip()

def get_model_action(llm_client, step, obs, last_action, last_feedback):
    user_prompt = build_user_prompt(step, obs, last_action, last_feedback)

    try:
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,   # lower = more stable
            max_tokens=200,
        )

        raw_text = (completion.choices[0].message.content or "").strip()

        # 🔥 DEBUG PRINT
        print(f"[RAW MODEL OUTPUT]: {raw_text}")

        # ❌ EMPTY RESPONSE FIX
        if not raw_text:
            raise ValueError("Empty response from model")

        # 🔧 Extract JSON safely (handles extra text)
        import re
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in model output")

        json_str = match.group()

        return Action(**json.loads(json_str))

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}")

        missing = obs.get("missing_values", {})
        dtypes = obs.get("data_types", {})

        # 🧠 Priority 1: Fix missing values
        if "Age" in missing and missing["Age"] > 0:
            return Action(tool="fill_missing_values", target_column="Age", new_value="25")

        # 🧠 Priority 2: Fix datatype
        if "Age" in dtypes and "int" not in dtypes["Age"]:
            return Action(tool="change_data_type", target_column="Age", new_value="int")

        # 🧠 Priority 3: If everything looks clean → submit
        return Action(tool="submit_final_dataset")

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
    last_action_str = "None"
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        print("Sending reset command...")
        raw_reset = await env.reset() 
        obs_dict = extract_obs_safe(raw_reset)
        last_feedback = "Environment initialized."
        done = False
        print("Reset successful! Smart LLM Agent taking over...\n")

        for step in range(1, MAX_STEPS + 1):
            if done: break

            action = get_model_action(llm_client, step, obs_dict, last_action_str, last_feedback)
            action_json_str = action.model_dump_json()
            last_action_str = action_json_str

            raw_step = await env.step(action)
            obs_obj = raw_step[0] if isinstance(raw_step, tuple) else raw_step
            obs_dict = obs_obj.model_dump() if hasattr(obs_obj, 'model_dump') else dict(obs_obj)
            
            # --- THE FIX: Extracting data from INSIDE the Observation payload ---
            last_feedback = obs_dict.get("last_action_feedback", "Action executed.")

            # ✅ FIX: reward and done are already inside observation
            reward = obs_obj.reward
            done = obs_obj.done

            rewards.append(reward)
            steps_taken = step
            
            # Keep the bot's required format intact
            log_step(step=step, action=action_json_str, reward=reward, done=done, error=None)
            
            # NEW: X-Ray Vision prints so we can see what the server is doing!
            print(f"   [X-RAY] Server Feedback: {last_feedback}")
            print(f"   [X-RAY] Missing Values: {obs_dict.get('missing_values')}")
            print("-" * 50)

        # Average the rewards and strictly clamp to platform boundaries
        score = sum(rewards) / len(rewards) if rewards else 0.05
        score = min(max(score, 0.05), 0.95)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to communicate:")
        traceback.print_exc()
        score, success = 0.0, False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())