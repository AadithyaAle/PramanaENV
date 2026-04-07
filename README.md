# 🧹 Meta OpenEnv: Self-Healing Data Pipeline Environment

**Built for the Meta PyTorch OpenEnv Hackathon**
By: Team [Your Team Name] (Aadithya Ale & Teammate)

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/YourUsername/YourSpaceName)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch Validated](https://img.shields.io/badge/PyTorch-Validated-EE4C2C.svg)](https://pytorch.org/)

---

## 🎯 Overview
Most LLM environments test if an agent can write code. We built an environment to test if an agent can **think like a Senior Data Engineer**. 

The **Data Cleaner Env** is a rigorous, state-managed OpenEnv sandbox where agents must sanitize heavily corrupted datasets. Rather than relying on rigid unit tests, this environment uses a **live PyTorch compilation check** as the ultimate judge of data cleanliness.

## ✨ Standout "Pro-Tier" Features

We designed this environment to punish lazy AI behavior and reward cost-effective, robust planning:

* 🧠 **Live PyTorch Validator:** Agents cannot fake success. Upon submission, the environment attempts to compile the numeric columns directly into a `torch.tensor`. If it fails due to hidden strings, commas, or nulls, the agent fails.
* 💸 **Economic Efficiency Penalty:** API calls aren't free. The environment applies a `-0.05` reward penalty for every action taken, forcing the agent to optimize its tool usage rather than blindly guessing.
* 🕰️ **Temporal Safety (Time-Travel):** Data pipelines are fragile. We implemented a state-history stack (`undo_last_action` tool) allowing agents to revert their own hallucinations and recover from mistakes without destroying the dataset.
* 🕵️ **Anti-Cheat Data Loss Monitor:** Agents are strictly monitored for data loss. If an agent lazily drops rows instead of cleaning them, the environment detects the length mismatch and fails the submission.
* 🦠 **Silent Corruption Edge Cases:** The "Hard" difficulty injects invisible whitespace and newlines (`" 4,000 \n"`) alongside commas, testing the agent's ability to handle real-world, filthy data.

---

## 🛠️ Environment Architecture

### State (Observation Space)
The agent receives a heavily typed Pydantic `Observation` containing:
* `current_columns`: List of active headers.
* `data_types`: Deeply inspected Pandas dtypes.
* `missing_values`: Null counts per column.
* `data_preview`: Markdown representation of the DataFrame `head()`.
* `last_action_feedback`: Narrative, self-correcting feedback (e.g., *"Cannot undo: No history available"*).

### Tools (Action Space)
* `drop_missing_rows`
* `fill_missing_values`
* `rename_column`
* `change_data_type`
* `undo_last_action`
* `submit_final_dataset`

---

## 🚀 Quickstart & Local Testing

### 1. Installation
Clone the repository and install the strict dependencies:
```bash
git clone [https://github.com/AadithyaAle/Meta-ENV-Hackathon.git](https://github.com/AadithyaAle/Meta-ENV-Hackathon.git)
cd Meta-ENV-Hackathon
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
HF_TOKEN=hf_your_hugging_face_token
API_BASE_URL=[https://router.huggingface.co/v1](https://router.huggingface.co/v1)
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

### 3. Run Inference
Watch the agent interact with the environment in real-time:
```bash
python3 inference.py
```

---

## 📊 Baseline Evaluation (Qwen2.5-72B-Instruct)

We evaluated the Qwen 72B model on the environment. By utilizing our custom instructional prompts and feedback loops, it successfully navigated the environment.

* **Task:** Medium (Rename columns, cast types, submit)
* **Steps Taken:** 3 (Rename -> Cast -> Submit)
* **Efficiency Penalties:** -0.10 (2 actions)
* **Submission Reward:** +1.00
* **Final Score:** **0.900** (Perfect execution)

*Agent successfully avoided hallucination loops by adhering to the `CRITICAL` state-instructions provided dynamically in the observation.*
```
