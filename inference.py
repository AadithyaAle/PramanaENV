import os
import json
import re
import asyncio
import sys
from openai import OpenAI
from openenv.core.env_client import EnvClient

# Force Python to find models.py in the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from models import Action

# 1. REQUIRED ENVIRONMENT VARIABLES
# Must include defaults for API_BASE_URL and MODEL_NAME
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Must NOT include a default for HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Target Environment URL
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://sukuna191552s-pramanaenv.hf.space")

# 2. OPENAI CLIENT REQUIREMENT
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# 3. STRICT OUTPUT FORMATTERS
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    err_str = "null" if error is None else str(error)
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err_str}", flush=True)

def log_end(success: bool, steps: int, rewards: list[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# 4. ROBUST AGENT LOGIC
def get_model_action(step: int, obs_dict: dict) -> Action:
    prompt = f"""
    Step: {step}
    Current Dataset State: {json.dumps(obs_dict, indent=2)}
    
    You are an expert Data Engineer. Analyze the target_schema_instructions and output the next logical action.
    Available tools: fill_missing_values, change_data_type, drop_missing_rows, submit_final_dataset.
    You must output strictly as a JSON object: {{"tool": "tool_name", "target_column": "ColName", "new_value": "val"}}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # Low temperature for stability within the 8GB RAM constraint
            max_tokens=200
        )
        raw_text = (response.choices[0].message.content or "").strip()
        
        # Safe JSON Extraction (prevents 'Expecting value' crashes)
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response.")
            
        return Action(**json.loads(match.group()))
    except Exception as e:
        # Failsafe: Undo if the API rate limits us, preventing an illegal submission loop
        return Action(tool="undo_last_action")

async def main():
    env_client = EnvClient(HF_SPACE_URL)
    
    # Required tracking variables
    task_name = "data_cleaning_challenge"
    env_name = "SST_hackathon_env"
    rewards = []
    steps_taken = 0
    success = False
    
    # Emit exact START line
    log_start(task=task_name, env=env_name, model=MODEL_NAME)
    
    try:
        raw_reset = await env_client.reset()
        obs_obj = raw_reset[0] if isinstance(raw_reset, tuple) else raw_reset
        obs_dict = obs_obj.model_dump() if hasattr(obs_obj, 'model_dump') else dict(obs_obj)
        
        done = False
        for step in range(1, 11):
            if done: break
                
            steps_taken = step
            action = get_model_action(step, obs_dict)
            
            # Condense JSON action to a single string line (No newlines allowed in stdout)
            action_str = action.model_dump_json(exclude_none=True).replace('\n', '')
            
            raw_step = await env_client.step(action)
            obs_obj = raw_step[0] if isinstance(raw_step, tuple) else raw_step
            obs_dict = obs_obj.model_dump() if hasattr(obs_obj, 'model_dump') else dict(obs_obj)
            
            # Extract properly using attributes
            reward = getattr(obs_obj, 'reward', 0.05)
            done = getattr(obs_obj, 'done', False)
            error_msg = None 
            
            rewards.append(reward)
            
            # Emit exact STEP line
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            
        # Determine final success (If the environment graded it > 0.90)
        if rewards and rewards[-1] >= 0.90:
            success = True
            
    except Exception as e:
        pass # Catch silently to ensure the mandatory [END] tag always prints
    finally:
        if not rewards:
            rewards = [0.00]
            steps_taken = 1
        # Emit exact END line (Removed the invalid 'score=' field)
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())