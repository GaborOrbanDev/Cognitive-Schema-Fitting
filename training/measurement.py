import sys
import os
import re
import time
import json
import pandas as pd
import pickle
import concurrent.futures as cf
from langchain_community.callbacks import get_openai_callback
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphRecursionError
from tqdm import tqdm
from pydantic import BaseModel, Field
from pydantic_core import ValidationError

# Add the parent directory to the sys path to import the schemas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.agent_state_classes import AgentInput, Task
from schemas.cot_sc_schema import CoTSCAgent
from schemas.cot_schema import CoTAgent
from schemas.cot_w_sr_schema import CoTwSRAgent
from schemas.spp_schema import SPPAgent
from schemas.tot_schema import ToTAgent


class Measurement(BaseModel):
    measurement_id: str = Field(..., description="ID of the measurement")
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
    task_text = f"**Question**: {task['question']} | **Answer choices**: {task['choices']}"
    choice_index = task["answer"]
    task_id = task["task_id"]
    subject_of_task = task["subject"]
        
    with get_openai_callback() as cb:
        start_time = time.perf_counter()
        
        try:
            response: dict[str, Task] = agent.invoke(AgentInput(task=Task(description=task_text)))
        except GraphRecursionError as ex:
            print(f"GraphRecursionError: {ex}")
            return
        
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        accuracy = 1 if response["task"].solution.index == choice_index else 0

        measurement = Measurement(
            measurement_id=f"{label}_{task_id}".lower(),
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
    
    with open(f"./training/measurements/{measurement.measurement_id}.json", "w") as f:
        record = measurement.model_dump()
        json.dump(record, f, indent=2)


def get_history(label: str) -> list[int]:
    history = []
    for filename in os.listdir("./training/measurements"):
        if filename.endswith(".json") and filename.startswith(label):
            id = re.search(r"\d+", filename).group(0)
            id = int(id)
            history.append(id)
    return history

def filter_indexes(all_indexes: list[int], history: list[int]) -> list[int]:
    return list(set(all_indexes) - set(history))

def main(schema_to_use: object, label: str) -> None:
    # Load the sample indexes
    with open("training/sample_indexes.pkl", "rb") as f:
        sample_indexes: list[int] = pickle.load(f)

    # Filter the sample indexes
    print(f"Total samples: {len(sample_indexes)}")
    sample_indexes = filter_indexes(sample_indexes, get_history(label.lower()))
    print(f"Remaining samples: {len(sample_indexes)}")

    # Create the agent
    agent: CompiledStateGraph = schema_to_use().create_agent()
    # agent: CompiledStateGraph = agent.with_retry(retry_if_exception_type=(Exception, ValidationError))

    # Measure the agent performance
    with cf.ThreadPoolExecutor(max_workers=7) as exec:
        list(
            tqdm(
                exec.map(
                    lambda idx: measure(data.iloc[idx], label, agent), 
                    sample_indexes
                ),
                total=len(sample_indexes)
            )
        )

if __name__ == "__main__":
    main(ToTAgent, label="ToT")