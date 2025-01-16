from pydantic import BaseModel, Field


class Solution(BaseModel):
    """Solution for the given task. Choose the right option index from the list of options. Index starts from 0"""

    scratchpad: str = Field(..., description="The scratchpad is for parsing the solution to solution index. You might leave it alone.")
    index: int


class Task(BaseModel):
    description: str
    solution: Solution | None = None


class AgentInput(BaseModel):
    long_term_goal: str = Field(default="Solve the task accurately and efficiently")
    task_history: list[Task] = []
    task: Task


class AgentOutput(BaseModel):
    task: Task