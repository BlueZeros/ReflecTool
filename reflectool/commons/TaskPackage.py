import time
import uuid

from pydantic import BaseModel
from typing import Optional


class TaskPackage(BaseModel):
    task_id: int = -1
    dataset: Optional[str] = None
    inputs: Optional[str] = None
    instruction: str
    multimodal_inputs: dict = {"image": None, "sql_database": None, "upload_files": None}
    creator:str = "human"
    completion: str = "active"
    answer: Optional[str] = None
    prev_act_obs: Optional[list] = []
    previous_answer: Optional[str] = None
    executor: str = ""
    ground_truth: str = ""
    eval: dict = {}

    def __str__(self):
        return f"""Task ID: {self.task_id}\nInputs: {self.inputs}\nInstruction: {self.instruction}\nTask Completion:{self.completion}\nAnswer: {self.answer}"""