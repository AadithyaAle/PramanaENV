from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional, Dict, List

class Observation(BaseModel):
    # This config line allows OpenEnv to be a bit more forgiving
    model_config = ConfigDict(extra='ignore') 
    
    current_columns: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    data_preview: str 
    target_schema_instructions: str
    last_action_feedback: str 

class Action(BaseModel):
    model_config = ConfigDict(extra='ignore')

    tool: Literal[
        "drop_missing_rows", 
        "fill_missing_values", 
        "rename_column", 
        "change_data_type", 
        "submit_final_dataset"
        "undo_last_action"
    ]
    target_column: Optional[str] = None
    new_value: Optional[str] = None