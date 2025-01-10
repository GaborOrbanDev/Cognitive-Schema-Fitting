# %% Importing libraries
import os
import openai
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# %% Schema classes

class Solution(BaseModel):
    """Solution for the given task. Choose the right option index from the list of options. Index starts from 0"""

    scratchpad: str = Field(..., description="The scratchpad parsing the solution to solution index. You might leave it alone.")
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


class AgentState(AgentInput, AgentOutput):
    messages: Annotated[list[AnyMessage], add_messages] = []


# %% Agent class
class CoTAgent:
    def __init__(self, temperature: float = 0.5, prompt_file_path: str | None = None) -> None:
        if prompt_file_path is None:
            prompt_file_path = "./prompts/cot_prompts.yaml"
        with open(prompt_file_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    def create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState, input=AgentInput, output=AgentOutput)
        workflow.add_node("schema_setup", self._schema_setup)
        workflow.add_node("cognition", self._cognition)
        workflow.add_node("resolve", self._resolve)
        workflow.add_edge(START, "schema_setup")
        workflow.add_edge("schema_setup", "cognition")
        workflow.add_edge("cognition", "resolve")
        workflow.add_edge("resolve", END)
        return workflow.compile()
    
    def __call__(self):
        return self.create_graph()
    
    # --------------------------------------------------------------------------------

    def _schema_setup(self, state: AgentInput) -> AgentState:
        prompt_template = SystemMessagePromptTemplate.from_template(self.prompts["system_prompt"])
        message = prompt_template.format(**{
            "long_term_goal": state.long_term_goal,
            "task_history": state.task_history,
            "task": state.task.description
        })
        return {"messages": [message]}
    
    def _cognition(self, state: AgentState) -> AgentState:
        return {"messages": [self.llm.invoke(state.messages)]}
    
    def _resolve(self, state: AgentState) -> AgentState:
        chain = (
            ChatPromptTemplate.from_messages([
                *state.messages,
                ("human", "{input}")
            ])
            | self.llm.with_structured_output(Solution)
        )
        solution = chain.invoke({"input": self.prompts["resolution_prompt"]})
        state.task.solution = solution
        return {"task": state.task}
    
# %% Testing the agent
graph = CoTAgent().create_graph()