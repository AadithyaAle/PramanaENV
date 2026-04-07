# OpenEnv: Messy Data Pipeline Cleaner 🧹📊

## Environment Description & Motivation
Data cleaning and preprocessing is universally recognized as the single largest bottleneck in the machine learning lifecycle. This environment models the genuine, highly valuable real-world task of an automated Data Engineer. 

An AI agent is provided with a corrupted dataset (containing missing values, incorrect data types, and wrong column names) and must use standard data engineering operations to transform it into a clean state that perfectly matches a target schema. This environment tests an LLM's ability to reason about data structures, execute precise tabular transformations, and recover from strict formatting errors.

## Action and Observation Spaces

This environment strictly follows the OpenEnv specification using typed Pydantic models.

### Observation Space
The agent receives a state dictionary representing the current condition of the dataset:
* `current_columns` (List[str]): The current column names in the dataset.
* `data_types` (Dict[str, str]): Dictionary mapping columns to their pandas `dtypes`.
* `missing_values` (Dict[str, int]): Count of `NaN`/Null values per column.
* `data_preview` (str): A markdown table of the first 5 rows to provide the LLM with contextual data grounding.
* `target_schema_instructions` (str): The goal state the agent must reach to achieve a reward of 1.0.
* `last_action_feedback` (str): System feedback from the previous action (e.g., "Success: Dropped 5 rows" or "Error: KeyError - column not found").

### Action Space
The agent must return a structured JSON object choosing a specific data engineering tool and its parameters:
* `tool` (Literal): The operation to perform. Must be one of:
  * `drop_missing_rows`
  * `fill_missing_values`
  * `rename_column`
  * `change_data_type`
  * `submit_final_dataset`
* `target_column` (Optional[str]): The specific column to apply the transformation to.
* `new_value` (Optional[str]): The new name (for renaming) or replacement value (for filling nulls/casting types).

## Task Descriptions & Difficulty

The environment features an automated, deterministic grader that evaluates the final dataset against a hidden "perfect" dataframe. 

1. **Task 1 (Easy):** * **Objective:** Clean a dataset containing simple null values.
   * **Expected Actions:** The agent must identify the column with missing values, use the `drop_missing_rows` tool to remove those specific rows, and submit the dataset.

2. **Task 2 (Medium):** * **Objective:** Standardize a dataset with bad metadata.
   * **Expected Actions:** The agent must use `rename_column` to fix improperly formatted column headers (e.g., changing `usr_nm` to `username`) and use `change_data_type` to cast string-based numerical columns into proper integer formats.

3. **Task 3 (Hard):** * **Objective:** Perform complex data imputation and string parsing.
   * **Expected Actions:** The agent must identify missing values that cannot be dropped, use `fill_missing_values` to safely impute them with a default string (e.g., "Unknown"), and parse complex numerical strings containing commas (e.g., "1,000") into valid integers without crashing the pipeline.

## Setup and Usage Instructions

### Prerequisites
* Python 3.10+
* Docker (for final validation and OpenEnv deployment)

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Install the required dependencies:**
   ```bash
   pip install pandas gymnasium openenv-core pydantic openai python-dotenv
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Hugging Face API token:
   ```env
   HF_TOKEN="your_hf_access_token_here"
   API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
   MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
   ```

### Running the Inference Baseline
To run the automated agent against the environment and reproduce the baseline scores:
```bash
python inference.py
```

### Running the OpenEnv Validator
To ensure the Hugging Face Space deployment and Docker container meet the hackathon requirements:
```bash
uvicorn server.app:app --reload
```

## Project Structure

```
SST_hackathon_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # SstHackathonEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── SST_hackathon_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
