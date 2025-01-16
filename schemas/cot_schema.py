"""
Chain of Thought cognitive schema implementation
source: https://arxiv.org/abs/2201.11903
"""

# %% Importing libraries
from __future__ import annotations
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
from schemas.agent_state_classes import AgentInput, AgentOutput, Solution

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# %% Schema state class
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

    def create_agent(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState, input=AgentInput, output=AgentOutput)
        workflow.add_node("schema_setup", self._schema_setup)
        workflow.add_node("cognition", self._cognition)
        workflow.add_node("resolution", self._resolution)
        workflow.add_edge(START, "schema_setup")
        workflow.add_edge("schema_setup", "cognition")
        workflow.add_edge("cognition", "resolution")
        workflow.add_edge("resolution", END)
        return workflow.compile()
    
    def __call__(self):
        return self.create_agent()
    
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
    
    def _resolution(self, state: AgentState) -> AgentOutput:
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
if __name__ == "__main__":
    graph = CoTAgent().create_agent()