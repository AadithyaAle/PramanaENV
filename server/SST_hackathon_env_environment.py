import pandas as pd
import numpy as np
import torch
import sys
import os
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action as SstHackathonAction
from models import Observation as SstHackathonObservation


# ── per-task grading functions ────────────────────────────────────────────────
# Each returns a float strictly in (0.0, 1.0) — never 0.0 or 1.0.

def _grade_task_1_age(df: pd.DataFrame) -> tuple[float, str]:
    """
    task_1_age: Fill missing Age with 25 AND cast to integer.
    Both conditions must be true for full marks.
    """
    if "Age" not in df.columns:
        return 0.1, "FAIL: 'Age' column is missing."

    has_nulls = df["Age"].isnull().any()
    is_int    = pd.api.types.is_integer_dtype(df["Age"])

    if has_nulls and not is_int:
        return 0.1, "FAIL: Age still has nulls and is not an integer type."
    if has_nulls:
        return 0.3, "PARTIAL: Age still has nulls (not filled yet)."
    if not is_int:
        return 0.3, "PARTIAL: Age nulls are filled but dtype is not integer."

    # Check the filled values are actually 25
    # Original dataset had NaN at index 2 and 4
    try:
        torch.tensor(df["Age"].values, dtype=torch.int32)
    except Exception as e:
        return 0.2, f"FAIL: PyTorch tensor conversion failed: {e}"

    return 0.95, "PASS: Age filled and cast to int successfully."


def _grade_task_2_salary(df: pd.DataFrame) -> tuple[float, str]:
    """
    task_2_salary: Drop rows where Salary is missing.
    """
    if "Salary" not in df.columns:
        return 0.1, "FAIL: 'Salary' column is missing."

    has_nulls = df["Salary"].isnull().any()
    if has_nulls:
        return 0.1, "FAIL: Salary still has null rows (not dropped)."

    try:
        torch.tensor(df["Salary"].values, dtype=torch.float32)
    except Exception as e:
        return 0.2, f"FAIL: PyTorch tensor conversion failed: {e}"

    return 0.95, "PASS: Salary rows with nulls dropped successfully."


def _grade_task_3_price(df: pd.DataFrame) -> tuple[float, str]:
    """
    task_3_price: Convert Price column to integer type.
    """
    if "Price" not in df.columns:
        return 0.1, "FAIL: 'Price' column is missing."

    is_int = pd.api.types.is_integer_dtype(df["Price"])
    if not is_int:
        return 0.2, f"FAIL: Price dtype is '{df['Price'].dtype}', expected integer."

    try:
        torch.tensor(df["Price"].values, dtype=torch.int32)
    except Exception as e:
        return 0.2, f"FAIL: PyTorch tensor conversion failed: {e}"

    return 0.95, "PASS: Price converted to integer successfully."


TASK_GRADERS = {
    "task_1_age":    _grade_task_1_age,
    "task_2_salary": _grade_task_2_salary,
    "task_3_price":  _grade_task_3_price,
}
# ─────────────────────────────────────────────────────────────────────────────


class SstHackathonEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._history: list = []          # for undo support

        self.tasks = [
            {
                "name": "task_1_age",
                "data": pd.DataFrame({
                    "usr_nm": ["Alice", "Bob", "Charlie", "David", "Eve"],
                    "Age":    ["25", "30", np.nan, "22", np.nan],
                }),
                "instructions": (
                    "Clean the dataset. Fill missing Age values with 25 "
                    "and cast the Age column to integer dtype."
                ),
            },
            {
                "name": "task_2_salary",
                "data": pd.DataFrame({
                    "Salary":     [50000, 60000, np.nan],
                    "Department": ["IT", "HR", "IT"],
                }),
                "instructions": (
                    "Clean the dataset. Drop all rows where Salary is missing."
                ),
            },
            {
                "name": "task_3_price",
                "data": pd.DataFrame({
                    "Price": ["10", "20", "30"],
                    "Item":  ["Apple", "Banana", "Cherry"],
                }),
                "instructions": (
                    "Clean the dataset. Change the Price column data type to int."
                ),
            },
        ]
        self.current_task_idx = 0
        self.current_task_name = ""
        self.df = pd.DataFrame()
        self.instructions = ""

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> SstHackathonObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._history = []
        task = self.tasks[self.current_task_idx]
        self.df = task["data"].copy()
        self.instructions = task["instructions"]
        self.current_task_name = task["name"]
        self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        return self._get_observation("Environment reset successfully.", 0.05, False)

    # ── observation builder ───────────────────────────────────────────────────
    def _get_observation(
        self, feedback: str, reward: float, done: bool
    ) -> SstHackathonObservation:
        # Clamp reward so it is NEVER exactly 0.0 or 1.0
        reward = float(min(max(reward, 0.05), 0.95))
        return SstHackathonObservation(
            current_columns=list(self.df.columns),
            data_types={col: str(dtype) for col, dtype in self.df.dtypes.items()},
            missing_values=self.df.isnull().sum().to_dict(),
            data_preview=(
                self.df.head().to_markdown() if not self.df.empty else ""
            ),
            last_action_feedback=feedback,
            target_schema_instructions=self.instructions,
            reward=reward,
            done=done,
        )

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: SstHackathonAction) -> SstHackathonObservation:
        self._state.step_count += 1
        reward = 0.05
        done = False
        feedback = ""

        try:
            tool = action.tool
            col  = action.target_column
            val  = action.new_value

            # ── fill_missing_values ──────────────────────────────────────────
            if tool == "fill_missing_values":
                if col not in self.df.columns:
                    feedback = f"Error: Column '{col}' not found."
                else:
                    self._history.append(self.df.copy())
                    before = self.df[col].isnull().sum()
                    self.df[col] = self.df[col].fillna(val)
                    after = self.df[col].isnull().sum()
                    filled = before - after
                    reward = 0.3 if filled > 0 else 0.05
                    feedback = (
                        f"Filled {filled} null(s) in '{col}' with '{val}'."
                        if filled > 0
                        else f"No nulls found in '{col}' to fill."
                    )

            # ── change_data_type ─────────────────────────────────────────────
            elif tool == "change_data_type":
                if col not in self.df.columns:
                    feedback = f"Error: Column '{col}' not found."
                else:
                    self._history.append(self.df.copy())
                    try:
                        self.df[col] = self.df[col].astype(val)
                        reward = 0.3
                        feedback = f"Changed '{col}' dtype to {val}."
                    except Exception as e:
                        self.df = self._history.pop()   # rollback
                        feedback = f"Type cast failed: {e}"

            # ── drop_missing_rows ────────────────────────────────────────────
            elif tool == "drop_missing_rows":
                if col not in self.df.columns:
                    feedback = f"Error: Column '{col}' not found."
                else:
                    self._history.append(self.df.copy())
                    before = len(self.df)
                    self.df = self.df.dropna(subset=[col])
                    dropped = before - len(self.df)
                    reward = 0.3 if dropped > 0 else 0.05
                    feedback = (
                        f"Dropped {dropped} row(s) with nulls in '{col}'."
                        if dropped > 0
                        else f"No null rows found in '{col}'."
                    )

            # ── undo_last_action ─────────────────────────────────────────────
            elif tool == "undo_last_action":
                if self._history:
                    self.df = self._history.pop()
                    reward = 0.05
                    feedback = "Undid the last action."
                else:
                    reward = 0.05
                    feedback = "Cannot undo: no history available."

            # ── rename_column ────────────────────────────────────────────────
            elif tool == "rename_column":
                if col not in self.df.columns:
                    feedback = f"Error: Column '{col}' not found."
                else:
                    self._history.append(self.df.copy())
                    self.df = self.df.rename(columns={col: val})
                    reward = 0.1
                    feedback = f"Renamed column '{col}' to '{val}'."

            # ── submit_final_dataset ─────────────────────────────────────────
            elif tool == "submit_final_dataset":
                done = True

                print("DEBUG: Submitting task:", self.current_task_name)
                print("DEBUG: Available graders:", TASK_GRADERS.keys())

                grader = TASK_GRADERS.get(self.current_task_name)

                if grader is None:
                    reward = 0.1
                    feedback = f"No grader found for task '{self.current_task_name}'."
                else:
                    reward, feedback = grader(self.df)
                    print("DEBUG: Grader result:", reward, feedback)

            else:
                feedback = f"Unknown tool '{tool}'."

        except Exception as e:
            feedback = f"Unexpected error: {e}"
            reward = 0.05

        # Hard step limit
        if self._state.step_count >= 10:
            done = True

        return self._get_observation(feedback, reward, done)

    # ── state property ────────────────────────────────────────────────────────
    @property
    def state(self) -> State:
        return self._state