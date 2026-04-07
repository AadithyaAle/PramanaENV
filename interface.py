import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

# TODO: Adjust this import to match whatever Teammate 1 named the file and classes!
from envs.data_cleaner_env import DataCleanerEnv, Action

IMAGE_NAME = os.getenv("IMAGE_NAME") 
# The hackathon script checks HF_TOKEN first
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Defaulting to the Hugging Face router as requested by the hackathon spec
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("TASK_NAME", "data_cleaning")
BENCHMARK = os.getenv("BENCHMARK", "openenv_data_cleaner")
MAX_STEPS = 10
TEMPERATURE = 0.1 # Low temperature for strict JSON generation
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.8 

# --- MANDATORY LOGGING FUNCTIONS (DO NOT TOUCH) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- AI AGENT LOGIC ---
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

def get_model_action(client: OpenAI, step: int, obs: dict, last_feedback: str) -> Action:
    user_prompt = build_user_prompt(step, obs, last_feedback)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            # Ensuring strict JSON output format
            response_format={"type": "json_object"} 
        )
        
        raw_text = (completion.choices[0].message.content or "").strip()
        
        # Parse the JSON string from the LLM into the Pydantic Action model
        action_dict = json.loads(raw_text)
        return Action(**action_dict)
        
    except Exception as exc:
        print(f"[DEBUG] Model request failed or hallucinated bad JSON: {exc}", flush=True)
        # Fallback to prevent the script from crashing if the LLM errors out
        return Action(tool="submit_final_dataset")

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Note: For local testing, use direct class initialization.
    # For final submission, ensure this uses await DataCleanerEnv.from_docker_image(IMAGE_NAME)
    # as specified in Meta's OpenEnv docs.
    env = DataCleanerEnv() 

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Await the reset if your environment is async, otherwise standard env.reset()
        result = env.reset() 
        if asyncio.iscoroutine(result):
            result = await result
            
        obs_dict = result.observation.model_dump()
        last_feedback = "Environment initialized."

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # 1. Ask the Hugging Face LLM what to do
            action = get_model_action(client, step, obs_dict, last_feedback)
            action_json_str = action.model_dump_json()

            # 2. Send the action to Teammate 1's backend logic
            step_result = env.step(action)
            if asyncio.iscoroutine(step_result):
                step_result = await step_result
            
            # 3. Update the state
            obs_dict = step_result.observation.model_dump()
            reward = step_result.reward or 0.0
            done = step_result.done
            
            # Catching logic errors from Teammate 1's backend
            error = None 
            last_feedback = obs_dict.get("last_action_feedback", "No feedback")

            rewards.append(reward)
            steps_taken = step

            # 4. Mandatory stdout log
            log_step(step=step, action=action_json_str, reward=reward, done=done, error=error)

            if done:
                break

        # Calculate final score based on accumulated rewards
        score = sum(rewards) 
        score = min(max(score, 0.0), 1.0) 
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            close_result = env.close()
            if asyncio.iscoroutine(close_result):
                await close_result
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())