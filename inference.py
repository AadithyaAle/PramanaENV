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

# --- BULLETPROOF CLIENT ---
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
TEMPERATURE = 0.1 
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.8 

# --- LOGGING ---
def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error): print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)
def log_end(success, steps, score, rewards): print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# --- AI AGENT LOGIC (THE BRAIN) ---
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an automated Data Engineer Agent. Your job is to clean a messy dataset.
    You will receive the current state of the dataframe (columns, null counts, datatypes) and instructions.
    
    You must choose ONE of the following tools to fix the data:
    - drop_missing_rows
    - fill_missing_values
    - rename_column
    - change_data_type
    - submit_final_dataset
    
    You must output your decision STRICTLY as a JSON object matching this schema:
    {
        "tool": "tool_name",
        "target_column": "ColumnName",
        "new_value": "OptionalNewValue"
    }
    Reply ONLY with valid JSON. Do not include markdown formatting like ```json.
    """
).strip()

def build_user_prompt(step: int, obs: dict, last_feedback: str) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Last Action Feedback: {last_feedback}
        
        Current Dataset State:
        {json.dumps(obs, indent=2)}
        
        Analyze the target_schema_instructions inside the state, and provide the JSON for your next action.
        """
    ).strip()

def get_model_action(llm_client: OpenAI, step: int, obs: dict, last_feedback: str) -> Action:
    user_prompt = build_user_prompt(step, obs, last_feedback)
    
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
        print(f"[DEBUG] Model request failed or hallucinated bad JSON: {exc}", flush=True)
        # Fallback to prevent crashing
        return Action(tool="submit_final_dataset", target_column="", new_value="")

# --- UNIVERSAL PARSER HELPER ---
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
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        print("Sending reset command...")
        raw_reset = await env.reset() 
        obs_dict = extract_obs_safe(raw_reset)
        
        last_feedback = "Environment initialized."
        done = False
        print("Reset successful! Environment is alive. LLM Agent taking over...")

        for step in range(1, MAX_STEPS + 1):
            if done: break

            # Ask the LLM what to do
            action = get_model_action(llm_client, step, obs_dict, last_feedback)

            # Send the action to Hugging Face
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
            last_feedback = obs_dict.get("last_action_feedback", "No feedback")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action.model_dump_json(), reward=reward, done=done, error=None)

        score = sum(rewards) 
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to communicate:")
        traceback.print_exc()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())