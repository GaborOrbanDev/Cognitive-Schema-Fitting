"""This module contains the definitions of different predefined 
data structures used within CSF system to create a common interface."""

from typing_extensions import Annotated
from langgraph.graph import add_messages
from langgraph.graph.message import AnyMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------------
# region Task definitions

class Task(BaseModel):
    """`Task` object repesents a general subproblem, decomoposed from the orginal `input_problem` of the user"""
    name: str = Field(..., description="Name of the task")
    description: str = Field(..., 
                             description="Essential description with details of the task, including the expected output", 
                             examples=[{"task": "what is the capital of France?", "expected_output": "city name"},
                                       {"task": "who many times does the 'r' letter appear in strawberry?", "expected_output": "number of r's"}])
    

class SolvedTask(Task):
    """`SolvedTask` object represents a task that has been solved by the system"""
    solution: str = Field(..., description="The solution of the task")

# endregion    
# ---------------------------------------------------------------------------------
# region Schema definitions for problem decomposition component

class ProblemDecompositionInput(BaseModel):
    input_problem: str = Field(..., description="The original problem statement given by the user")


class ProblemDecompositionOutput(BaseModel):
    tasks: list[Task] = Field([], description="List of tasks decomposed from the original problem statement")


class AgentMessagesState(ProblemDecompositionInput, ProblemDecompositionOutput):
    messages: Annotated[list[AnyMessage], add_messages] = Field([])

# endregion
# ---------------------------------------------------------------------------------
# region Definitions for cognitive schema selection

class ExampleSelection(BaseModel):
    """This class represents the basic structure of example schema selection to support few-shot learning"""

    prompt: str = Field(..., description="General subproblem statement to be solved")
    schema_label: str = Field(..., description="The label of the selected cognitive schema to solve the problem")


class CognitiveSchema(BaseModel):
    name: str = Field(..., description="Name of the cognitive schema")
    label: str = Field(..., description="Label of the cognitive schema")
    description: str = Field(..., description="Description of the cognitive schema")
    examples: list[ExampleSelection] = Field([], description="List of examples to support few-shot learning")

# endregion
# ---------------------------------------------------------------------------------
# region State definitions for dedicated cognitive schema agents

class SchemaAgentInput(BaseModel):
    input_problem: str = Field(..., description="The original problem statement given by the user")
    task: Task = Field(..., description="The current task to be solved")
    task_history: list[SolvedTask] = Field([], description="List of tasks that have been solved so far")


class SchemaAgentOutput(BaseModel):
    solution: str = Field("<<SOLUTION NOT FOUND>>", description="The solution of the task without the thought process and the explanation. If the solution is not found, the value is `<<SOLUTION NOT FOUND>>`")

# endregion