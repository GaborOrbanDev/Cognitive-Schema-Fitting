import sys
import os
import time
import pandas as pd
from langchain_community.callbacks import get_openai_callback
from langgraph.graph.state import CompiledStateGraph
from tqdm import tqdm
from pydantic import BaseModel, Field

# Add the parent directory to the sys path to import the schemas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.agent_state_classes import AgentInput, Task
from schemas.cot_sc_schema import CoTSCAgent
from schemas.cot_schema import CoTAgent
from schemas.cot_w_sr_schema import CoTwSRAgent
from schemas.spp_schema import SPPAgent
from schemas.tot_schema import ToTAgent


class Measurement(BaseModel):
    cognitive_schema: str = Field(..., description="Label of the used cognitive schema")
    task_id: int = Field(..., description="ID of the task")
    task: str = Field(..., description="Text of the task")
    subject_of_task: str = Field(..., description="Subject of the task")
    accuracy: int = Field(..., description="Accuracy of the cognitive schema", ge=0, le=1)
    inference_time: float = Field(..., description="Inference time of the cognitive schema")
    total_tokens: int = Field(..., description="Total tokens used in the inference")
    prompt_tokens: int = Field(..., description="Tokens used in the prompt")
    completion_tokens: int = Field(..., description="Tokens used in the completion")
    cost: float = Field(..., description="Cost of the inference")


data = pd.read_csv("./training/mmlu_all_w_id.csv", sep=";")

def measure(task: dict, label: str, agent: CompiledStateGraph) -> None:
    task_text = f"**Question**: {task["question"]} | **Answer choices**: {task["choices"]}"
    choice_index = task["answer"]
    task_id = task["task_id"]
    subject_of_task = task["subject_of_task"]
        
    with get_openai_callback() as cb:
        start_time = time.perf_counter()
        response: dict[str, Task] = agent.invoke(AgentInput(task=Task(description=task_text)))
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        accuracy = 1 if response["task"].solution.index == choice_index else 0

        measurement = Measurement(
            cognitive_schema=label,
            task_id=task_id,
            task=task_text,
            subject_of_task=subject_of_task,
            accuracy=accuracy,
            inference_time=inference_time,
            total_tokens=cb.total_tokens,
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
            cost=cb.total_cost
        )

    