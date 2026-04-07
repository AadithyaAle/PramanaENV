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

        # Placeholders for gym API
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action_feedback = ""

        # Randomly select task: easy, medium, or hard
        self.current_task = random.choice(["easy", "medium", "hard"])

        # ---- TASK 1: EASY ----
        if self.current_task == "easy":
            self.df = pd.DataFrame({
                "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "Email": ["alice@example.com", "bob@example.com", None, "david@example.com", "eve@example.com"],
            })

        # ---- TASK 2: MEDIUM ----
        elif self.current_task == "medium":
            self.df = pd.DataFrame({
                "usr_nm": ["alice", "bob", "charlie", "david", "eve"],
                "Age": ["25", "30", "22", "40", "28"],
            })

        # ---- TASK 3: HARD ----
        elif self.current_task == "hard":
            self.df = pd.DataFrame({
                "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "Salary": ["1,000", "2,500", None, "4,000", None],
            })

        return self._get_observation(), {}

    def step(self, action: Action):
        reward = 0.0
        terminated = False
        truncated = False
        self.last_action_feedback = ""

        try:
            # ---- DROP MISSING ROWS ----
            if action.tool == "drop_missing_rows":
                if self.current_task == "hard":
                    raise ValueError("Dropping rows is not allowed in Hard task")

                target_column = action.target_column
                if target_column not in self.df.columns:
                    raise ValueError(f"Column {target_column} not found")

                self.df = self.df.dropna(subset=[target_column]).reset_index(drop=True)
                self.last_action_feedback = f"Dropped rows with missing values in {target_column}"

            # ---- FILL MISSING VALUES ----
            elif action.tool == "fill_missing_values":
                target_column = action.target_column
                new_value = action.new_value

                if target_column not in self.df.columns:
                    raise ValueError(f"Column {target_column} not found")

                self.df[target_column] = self.df[target_column].fillna(new_value)
                self.last_action_feedback = f"Filled missing values in {target_column}"

            # ---- RENAME COLUMN ----
            elif action.tool == "rename_column":
                target_column = action.target_column
                new_value = action.new_value

                if target_column not in self.df.columns:
                    raise ValueError(f"Column {target_column} not found")

                self.df = self.df.rename(columns={target_column: new_value})
                self.last_action_feedback = f"Renamed {target_column} to {new_value}"

            # ---- CHANGE DATA TYPE ----
            elif action.tool == "change_data_type":
                target_column = action.target_column
                new_value = action.new_value

                if target_column not in self.df.columns:
                    raise ValueError(f"Column {target_column} not found")

                # Handle comma strings if casting to int (Hard task)
                if new_value == "int":
                    self.df[target_column] = self.df[target_column].astype(str).str.replace(",", "")
                    self.df[target_column] = self.df[target_column].astype(int)
                elif new_value == "float":
                    self.df[target_column] = self.df[target_column].astype(float)
                elif new_value == "datetime":
                    self.df[target_column] = pd.to_datetime(self.df[target_column])
                else:
                    raise ValueError(f"Unsupported type {new_value}")

                self.last_action_feedback = f"Converted {target_column} to {new_value}"

            # ---- SUBMIT FINAL DATASET ----
            elif action.tool == "submit_final_dataset":
                if self.current_task == "easy":
                    reward = 1.0 if self.df["Email"].isna().sum() == 0 else 0.0

                elif self.current_task == "medium":
                    correct_columns = set(["username", "Age"])
                    columns_correct = set(self.df.columns) == correct_columns
                    dtype_correct = False
                    if "Age" in self.df.columns:
                        dtype_correct = pd.api.types.is_integer_dtype(self.df["Age"])
                    reward = 1.0 if (columns_correct and dtype_correct) else 0.0

                elif self.current_task == "hard":
                    no_missing = self.df["Salary"].isna().sum() == 0
                    dtype_correct = pd.api.types.is_integer_dtype(self.df["Salary"])
                    reward = 1.0 if (no_missing and dtype_correct) else 0.0

                terminated = True
                self.last_action_feedback = "Submitted final dataset"

            else:
                raise ValueError(f"Unknown tool {action.tool}")

        except Exception as e:
            reward = -0.1
            self.last_action_feedback = f"Error: {str(e)}"

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        instructions = "CRITICAL: If the column 'username' already exists and is an 'int', do NOT rename it back. Immediately use 'submit_final_dataset' to finish."
        if self.current_task == "easy":
            instructions = "Ensure there are no missing values in the dataset. Use drop_missing_rows. When finished, you MUST use the 'submit_final_dataset' tool."
        elif self.current_task == "medium":
            instructions = "Rename the column 'usr_nm' to 'username'. Ensure the 'Age' column is cast to an 'int'. When finished, you MUST use the 'submit_final_dataset' tool."
        elif self.current_task == "hard":
            instructions = "Do NOT drop rows. Fill missing 'Salary' values with '0'. Remove commas from the 'Salary' strings and convert the column to an 'int'. When finished, you MUST use the 'submit_final_dataset' tool."

        return Observation(
            current_columns=list(self.df.columns),
            # Force standard Python strings and ints to make Pydantic happy
            data_types={str(col): str(dtype) for col, dtype in self.df.dtypes.items()},
            missing_values={str(col): int(val) for col, val in self.df.isna().sum().items()},
            data_preview=self.df.head().to_markdown(),
            target_schema_instructions=instructions,
            last_action_feedback=self.last_action_feedback,
        )

    def render(self):
        """Print the current DataFrame and feedback for debugging"""
        print(f"\nTask: {self.current_task}")
        print(self.df)
        print("Last action feedback:", self.last_action_feedback)