# %% Importing libraries
import os
import uuid
from operator import add
from typing_extensions import Literal, TypedDict, Annotated
import yaml
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolCall
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import AnyMessage, add_messages
import openai
from pydantic import BaseModel, Field

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
class CoTwSRAgent:
    def __init__(self, temperature: float = 0.5, prompt_file_path: str | None = None, max_eval_count: int = 3) -> None:
        if prompt_file_path is None:
            prompt_file_path = "./prompts/cot_w_sr_prompts.yaml"
        with open(prompt_file_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        self.eval_counter = 0
        self.max_eval_count = max_eval_count

    def create_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState, input=AgentInput, output=AgentOutput)
        workflow.add_node("schema_setup", self._schema_setup)
        workflow.add_node("cognition", self._cognition)
        workflow.add_node("evaluation", self._evaluation)
        workflow.add_node("refinement", self._refinement)
        workflow.add_node("resolution", self._resolution)
        workflow.add_edge(START, "schema_setup")
        workflow.add_edge("schema_setup", "cognition")
        workflow.add_edge("cognition", "evaluation")
        workflow.add_conditional_edges("evaluation", self._refinement_router)
        workflow.add_edge("refinement", "evaluation") # Loop back to cognition if refinement was requested
        workflow.add_edge("resolution", END)
        return workflow.compile()
    
    def __call__(self):
        return self.create_graph()

    # --------------------------------------------------------------------------------

    def _schema_setup(self, state: AgentInput) -> AgentState:
        self.eval_counter = 0
        prompt_template = SystemMessagePromptTemplate.from_template(self.prompts["system_prompt"])
        message = prompt_template.format(**{
            "long_term_goal": state.long_term_goal,
            "task_history": state.task_history,
            "task": state.task.description
        })
        return {"messages": [message]}
    
    def _cognition(self, state: AgentState) -> AgentState:
        return {"messages": [self.llm.invoke(state.messages)]}
    
    def _get_prefix(self, caller: Literal["evaluation", "refinement"]) -> AIMessage:
        if caller == "evaluation":
            self.eval_counter += 1
        return AIMessage(f"============= {caller.capitalize()} {self.eval_counter} =============\n")
    
    def _evaluation(self, state: AgentState) -> AgentState:
        class SelfEvaluation(BaseModel):
            """Evalation structure for the agent"""
            evaluation: str = Field(..., description="The text of the evaluation")
            decision: Literal["resolution", "refinement"] = Field(..., description="The decision made by the agent based on the evaluation. Resoltion means the evaluation is good enough to resolve the task.")

        eval_llm = self.llm.with_structured_output(SelfEvaluation)
        evaluation = eval_llm.invoke(state.messages + [HumanMessage(self.prompts["evaluation_prompt"])])
        reponse = AIMessage(
            content=getattr(evaluation, "evaluation"),
            decision=getattr(evaluation, "decision")
        )
        return {"messages": [self._get_prefix(caller="evaluation"), reponse]}
    
    def _refinement(self, state: AgentState) -> AgentState:
        prefix = self._get_prefix(caller="refinement")
        response = self.llm.invoke(state.messages + [prefix, HumanMessage(self.prompts["refinement_prompt"])])
        return {"messages": [prefix, response]}

    def _refinement_router(self, state: AgentState) -> Literal["resolution", "refinement"]:
        decision: str | None = getattr(state.messages[-1], "decision", None)       
        if self.eval_counter > self.max_eval_count:
            return "resolution"
        elif isinstance(decision, str) and decision in ["resolution", "refinement"]:
            return decision
        else:
            return "refinement"
        
    def _resolution(self, state: AgentState) -> AgentState:
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
graph = CoTwSRAgent().create_graph()