import pandas as pd
import numpy as np
import torch
import sys
import os
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Force path to find models.py in the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action as SstHackathonAction
from models import Observation as SstHackathonObservation

class SstHackathonEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        self.tasks = [
            {
                "name": "task_1_age",
                "data": pd.DataFrame({"usr_nm": ["Alice", "Bob", "Charlie", "David", "Eve"], "Age": ["25", "30", np.nan, "22", np.nan]}),
                "instructions": "Clean the dataset. Fill missing Age with 25 and make it an int."
            },
            {
                "name": "task_2_salary",
                "data": pd.DataFrame({"Salary": [50000, 60000, np.nan], "Department": ["IT", "HR", "IT"]}),
                "instructions": "Clean the dataset. Drop rows with missing Salary."
            },
            {
                "name": "task_3_price",
                "data": pd.DataFrame({"Price": ["10", "20", "30"], "Item": ["Apple", "Banana", "Cherry"]}),
                "instructions": "Clean the dataset. Change Price data type to int."
            }
        ]
        self.current_task_idx = 0
        self.df = pd.DataFrame()
        self.instructions = ""

    def reset(self) -> SstHackathonObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        task = self.tasks[self.current_task_idx]
        self.df = task["data"].copy()
        self.instructions = task["instructions"]
        self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        return self._get_observation("Environment Reset successfully.", 0.0, False)

    def _get_observation(self, feedback: str, reward: float, done: bool) -> SstHackathonObservation:
        return SstHackathonObservation(
            current_columns=list(self.df.columns),
            data_types={col: str(dtype) for col, dtype in self.df.dtypes.items()},
            missing_values=self.df.isnull().sum().to_dict(),
            data_preview=self.df.head().to_markdown() if not self.df.empty else "",
            last_action_feedback=feedback,
            target_schema_instructions=self.instructions,
            reward=reward,
            done=done
        )

    def step(self, action: SstHackathonAction) -> SstHackathonObservation:
        self._state.step_count += 1
        reward = 0.0
        done = False
        feedback = ""

        try:
            if action.tool == "fill_missing_values" and action.target_column in self.df.columns:
                self.df[action.target_column] = self.df[action.target_column].fillna(action.new_value)
                reward = 0.5 
                feedback = f"Success: Filled nulls in {action.target_column}."
                
            elif action.tool == "change_data_type" and action.target_column in self.df.columns:
                self.df[action.target_column] = self.df[action.target_column].astype(action.new_value)
                reward = 0.5
                feedback = f"Success: Changed {action.target_column} type."
                
            elif action.tool == "drop_missing_rows" and action.target_column in self.df.columns:
                self.df = self.df.dropna(subset=[action.target_column])
                reward = 0.5
                feedback = f"Success: Dropped rows."
                
            elif action.tool == "submit_final_dataset":
                done = True
                reward = 0.5
                feedback = "Dataset submitted."
            else:
                feedback = f"Tool '{action.tool}' executed."

        except Exception as e:
            feedback = f"Action Error: {str(e)}"

        if self._state.step_count >= 10:
            done = True

        return self._get_observation(feedback, reward, done)

    @property
    def state(self) -> State:
        return self._state