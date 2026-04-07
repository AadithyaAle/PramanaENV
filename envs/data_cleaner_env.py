# envs/data_cleaner_env.py

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import random

from models import Observation, Action


class DataCleanerEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.df = None
        self.current_task = None
        self.last_action_feedback = ""

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.last_action_feedback = ""

        # Randomly choose task
        self.current_task = random.choice(["easy", "medium", "hard"])

        # ---- TASK 1: EASY ----
        if self.current_task == "easy":
            data = {
                "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "Email": [
                    "alice@example.com",
                    "bob@example.com",
                    None,
                    "david@example.com",
                    "eve@example.com",
                ],
            }
            self.df = pd.DataFrame(data)

        # ---- TASK 2: MEDIUM ----
        elif self.current_task == "medium":
            data = {
                "usr_nm": ["alice", "bob", "charlie", "david", "eve"],
                "Age": ["25", "30", "22", "40", "28"],
            }
            self.df = pd.DataFrame(data)

        # ---- TASK 3: HARD ----
        elif self.current_task == "hard":
            data = {
                "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "Salary": ["1,000", "2,500", None, "4,000", None],  # commas + missing
            }
            self.df = pd.DataFrame(data)

        return self._get_observation(), {}

    def step(self, action: Action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        tool = action.tool_name
        args = action.arguments

        # Reset feedback each step
        self.last_action_feedback = ""

        try:
            # ---- DROP MISSING ROWS ----
            if tool == "drop_missing_rows":
                if self.current_task == "hard":
                    raise ValueError("Dropping rows is not allowed in this task")

                target_column = args.get("target_column")

                if target_column not in self.df.columns:
                    raise ValueError(f"Column {target_column} not found")

                self.df = self.df.dropna(subset=[target_column]).reset_index(drop=True)
                self.last_action_feedback = "Dropped rows with missing values"

            # ---- RENAME COLUMN ----
            elif tool == "rename_column":
                target_column = args.get("target_column")
                new_value = args.get("new_value")

                if target_column not in self.df.columns:
                    raise ValueError(f"Column {target_column} not found")

                self.df = self.df.rename(columns={target_column: new_value})
                self.last_action_feedback = f"Renamed {target_column} to {new_value}"

            # ---- CHANGE DATA TYPE ----
            elif tool == "change_data_type":
                target_column = args.get("target_column")
                new_value = args.get("new_value")

                if target_column not in self.df.columns:
                    raise ValueError(f"Column {target_column} not found")

                # Special cleaning for comma numbers (Task 3)
                if new_value == "int":
                    self.df[target_column] = (
                        self.df[target_column]
                        .astype(str)
                        .str.replace(",", "")
                    )
                    self.df[target_column] = self.df[target_column].astype(int)

                elif new_value == "float":
                    self.df[target_column] = self.df[target_column].astype(float)

                elif new_value == "datetime":
                    self.df[target_column] = pd.to_datetime(self.df[target_column])

                else:
                    raise ValueError(f"Unsupported type {new_value}")

                self.last_action_feedback = f"Converted {target_column} to {new_value}"

            # ---- FILL MISSING VALUES ----
            elif tool == "fill_missing_values":
                target_column = args.get("target_column")
                new_value = args.get("new_value")

                if target_column not in self.df.columns:
                    raise ValueError(f"Column {target_column} not found")

                self.df[target_column] = self.df[target_column].fillna(new_value)
                self.last_action_feedback = f"Filled missing values in {target_column}"

            # ---- SUBMIT FINAL DATASET ----
            elif tool == "submit_final_dataset":

                # ---- EASY ----
                if self.current_task == "easy":
                    reward = 1.0 if self.df["Email"].isna().sum() == 0 else 0.0

                # ---- MEDIUM ----
                elif self.current_task == "medium":
                    correct_columns = set(["username", "Age"])
                    columns_correct = set(self.df.columns) == correct_columns

                    dtype_correct = False
                    if "Age" in self.df.columns:
                        dtype_correct = pd.api.types.is_integer_dtype(self.df["Age"])

                    reward = 1.0 if (columns_correct and dtype_correct) else 0.0

                # ---- HARD ----
                elif self.current_task == "hard":
                    no_missing = self.df["Salary"].isna().sum() == 0

                    dtype_correct = pd.api.types.is_integer_dtype(self.df["Salary"])

                    reward = 1.0 if (no_missing and dtype_correct) else 0.0

                terminated = True
                self.last_action_feedback = "Submission evaluated"

            else:
                raise ValueError(f"Unknown tool {tool}")

        # ---- GLOBAL ERROR HANDLER (CRITICAL FOR LLM SAFETY) ----
        except Exception as e:
            reward = -0.1
            self.last_action_feedback = f"Error: {str(e)}"

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return Observation(
            data=self.df.to_dict(orient="records"),
            columns=list(self.df.columns),
            # 👇 Add feedback for agent reasoning (VERY useful)
            last_action_feedback=self.last_action_feedback,
        )

    def render(self):
        print(f"\nTask: {self.current_task}")
        print(self.df)
        print("Feedback:", self.last_action_feedback)