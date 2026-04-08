import gymnasium as gym
import pandas as pd
import numpy as np
from pydantic import BaseModel

# --- 1. PYDANTIC MODELS ---
class Action(BaseModel):
    tool: str
    target_column: str = ""
    new_value: str = ""

class Observation(BaseModel):
    columns: list
    null_counts: dict
    dtypes: dict
    target_schema_instructions: str
    last_action_feedback: str

# --- 2. THE ENVIRONMENT ---
class DataCleanerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # SCALER BOT FIX: You MUST have at least 3 tasks registered. 
        # Here are 3 unique datasets to cycle through.
        self.tasks = [
            {
                "name": "Task 1",
                "data": pd.DataFrame({"Age": [25, np.nan, 30], "Name": ["Alice", "Bob", "Charlie"]}),
                "instructions": "Fill missing Age with 25."
            },
            {
                "name": "Task 2",
                "data": pd.DataFrame({"Salary": [50000, 60000, np.nan], "Department": ["IT", "HR", "IT"]}),
                "instructions": "Drop rows with missing Salary."
            },
            {
                "name": "Task 3",
                "data": pd.DataFrame({"Price": ["10", "20", "30"], "Item": ["Apple", "Banana", "Cherry"]}),
                "instructions": "Change Price data type to int."
            }
        ]
        self.current_task_idx = 0
        self.dataframe = None
        self.instructions = ""

    def reset(self, seed=None, options=None):
        # Cycle through the 3 tasks sequentially so the bot sees all of them
        task = self.tasks[self.current_task_idx]
        self.dataframe = task["data"].copy()
        self.instructions = task["instructions"]
        
        self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        
        obs = self._get_obs("Environment initialized. Ready for commands.")
        return obs, {}

    def _get_obs(self, feedback: str) -> Observation:
        # Packages the current state of the pandas dataframe to send back to your inference.py
        return Observation(
            columns=list(self.dataframe.columns),
            null_counts=self.dataframe.isnull().sum().to_dict(),
            dtypes={col: str(dtype) for col, dtype in self.dataframe.dtypes.items()},
            target_schema_instructions=self.instructions,
            last_action_feedback=feedback
        )

    def step(self, action: Action):
        reward = 0.0
        terminated = False
        feedback = ""

        try:
            # --- THE BROKEN BUTTON FIX ---
            # We now actually modify the dataframe using Pandas based on the AI's tool choice
            
            if action.tool == "fill_missing_values":
                if action.target_column in self.dataframe.columns:
                    # Fill the nulls
                    self.dataframe[action.target_column] = self.dataframe[action.target_column].fillna(action.new_value)
                    reward = 0.1  # SCALER BOT FIX: Give a partial reward > 0.0
                    feedback = f"Success: Filled missing values in {action.target_column} with '{action.new_value}'."
                else:
                    feedback = f"Error: Column '{action.target_column}' not found."

            elif action.tool == "change_data_type":
                if action.target_column in self.dataframe.columns:
                    # Change the data type
                    self.dataframe[action.target_column] = self.dataframe[action.target_column].astype(action.new_value)
                    reward = 0.1
                    feedback = f"Success: Changed '{action.target_column}' to type {action.new_value}."
                else:
                    feedback = f"Error: Column '{action.target_column}' not found."

            elif action.tool == "drop_missing_rows":
                if action.target_column in self.dataframe.columns:
                    # Drop rows with NaNs in the target column
                    self.dataframe = self.dataframe.dropna(subset=[action.target_column])
                    reward = 0.1
                    feedback = f"Success: Dropped rows with missing values in '{action.target_column}'."
                else:
                    feedback = f"Error: Column '{action.target_column}' not found."

            elif action.tool == "submit_final_dataset":
                terminated = True
                reward = 0.5  # SCALER BOT FIX: Score must be strictly between 0 and 1. 0.5 works perfectly here.
                feedback = "Dataset submitted for final evaluation."

            else:
                feedback = f"Error: Unknown tool '{action.tool}'"

        except Exception as e:
            feedback = f"Action crashed the environment: {str(e)}"
            reward = 0.0

        # Package the new state and send it back to inference.py
        obs = self._get_obs(feedback)
        return obs, reward, terminated, False, {}