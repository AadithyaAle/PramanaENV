# server/graders.py
#
# These are called by openenv.yaml if the platform supports the
# "grader: type: function" spec. They receive whatever state object
# the platform passes and return a float strictly in (0.0, 1.0).
#
# NOTE: The real grading logic lives inside SST_hackathon_env_environment.py
# in the submit_final_dataset branch. These functions are a safety net
# in case the platform calls them directly.

import pandas as pd


def _score_from_df(df, check_fn) -> float:
    """Helper: run a check function against a dataframe extracted from state."""
    try:
        if hasattr(state := df, "df"):           # state object with .df attribute
            actual_df = state.df
        elif isinstance(df, pd.DataFrame):
            actual_df = df
        else:
            return 0.5                           # unknown state shape — neutral score
        score, _ = check_fn(actual_df)
        return float(min(max(score, 0.05), 0.95))
    except Exception:
        return 0.5


def grade_task_1_age(state) -> float:
    """Grader for task_1_age: Fill missing Age with 25 and cast to int."""
    try:
        df = state.df if hasattr(state, "df") else None
        if df is None:
            return 0.5
        has_nulls = df["Age"].isnull().any()
        is_int = pd.api.types.is_integer_dtype(df["Age"])
        if not has_nulls and is_int:
            return 0.95
        if not has_nulls or is_int:
            return 0.3
        return 0.1
    except Exception:
        return 0.5


def grade_task_2_salary(state) -> float:
    """Grader for task_2_salary: Drop rows where Salary is missing."""
    try:
        df = state.df if hasattr(state, "df") else None
        if df is None:
            return 0.5
        has_nulls = df["Salary"].isnull().any()
        return 0.95 if not has_nulls else 0.1
    except Exception:
        return 0.5


def grade_task_3_price(state) -> float:
    """Grader for task_3_price: Convert Price column to integer type."""
    try:
        df = state.df if hasattr(state, "df") else None
        if df is None:
            return 0.5
        is_int = pd.api.types.is_integer_dtype(df["Price"])
        return 0.95 if is_int else 0.2
    except Exception:
        return 0.5