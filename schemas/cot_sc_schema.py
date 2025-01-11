# %% Importing libraries
import os
from pprint import pprint
import operator
from typing_extensions import Literal
import openai
from typing import Annotated, Any
import yaml
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# %% Schema classes
class Solution(BaseModel):
    """Solution for the given task. Choose the right option index from the list of options. Index starts from 0"""

    scratchpad: str = Field(..., description="The scratchpad is for parsing the solution to solution index. You might leave it alone.")
    index: int


class Task(BaseModel):
    description: str
    solution: Solution | None = None


class SampleResponse(BaseModel):
    """A sample response from the model for the given prompt"""

    chain_of_thought: str = Field(..., description="This is a scratchpad for the model to think on the question step by step")
    answer: str = Field(..., description="Here the model places the answer for the given question")


class AgentInput(BaseModel):
    long_term_goal: str = Field(default="Solve the task accurately and efficiently")
    task_history: list[Task] = []
    task: Task


class AgentOutput(BaseModel):
    task: Task


class AgentState(AgentInput, AgentOutput):
    messages: Annotated[list[AnyMessage], add_messages] = []
    samples: list[SampleResponse] = []
    largest_groups: list[list[SampleResponse]] = []


# %% Agent class
class CoTSCAgent:
    def __init__(self, temperature: float = 1, prompt_file_path: str | None = None, sample_count: int = 8) -> None:
        if prompt_file_path is None:
            prompt_file_path = "./prompts/cot_sc_prompts.yaml"
        with open(prompt_file_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        self.llm_aggregator = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        self.sample_count = sample_count

    def create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState, input=AgentInput, output=AgentOutput)
        workflow.add_node("schema_setup", self._schema_setup)
        workflow.add_node("sample_llm", self._sample_llm)
        workflow.add_node("aggregate_samples", self._aggregate_samples)
        workflow.add_node("choose_best_group", self._choose_best_group)
        workflow.add_node("resolution", self._resolution)
        workflow.add_edge(START, "schema_setup")
        workflow.add_edge("schema_setup", "sample_llm")
        workflow.add_edge("sample_llm", "aggregate_samples")
        workflow.add_conditional_edges("aggregate_samples", self._resolution_router)
        workflow.add_edge("choose_best_group", "resolution")
        workflow.add_edge("resolution", END)
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
    
    def _sample_llm(self, state: AgentState) -> AgentState:
        structured_llm = self.llm.with_structured_output(SampleResponse)
        # sampling the model
        samples: list[SampleResponse] = structured_llm.batch([
            state.messages
            for _ in range(self.sample_count)
        ])
        # creating messages
        messages = [
            sample_w_index
            for i, sample in enumerate(samples)
            for sample_w_index in 
            (
                AIMessage(f"========= Sample response index: {i} =========="),
                AIMessage(sample.answer)
            )
        ]
        return {"messages": messages, "samples": samples}
    
    def _aggregate_samples(self, state: AgentState) -> AgentState:
        # region method specific stuctures
        class SampleGroup(BaseModel):
            sample_indexies: list[int] = Field([], description="List of samples that are similar to each other. Each item of is the sample index in the original list of samples")
        
        class SampleAggregator(BaseModel):
            scratchpad: str = Field("", description="Scratchpad for the model where it can think on the similarity of the different samples before grouping them")
            sample_groups: list[SampleGroup] = Field([], description="List of groups of samples that are similar to each other")
            
            def get_largest_groups(self) -> list[list[SampleResponse]]:
                # getting the length of the largest group
                length = max(len(group.sample_indexies) for group in self.sample_groups)
                # returning the largest group(s)
                return [
                    [
                        state.samples[i] 
                        for i in group.sample_indexies
                    ] 
                    for group in self.sample_groups 
                    if len(group.sample_indexies) == length
                ]
        # endregion
        # region method logic    
        structured_llm = (
                            self.llm_aggregator
                            .with_structured_output(SampleAggregator)
                            .with_retry(retry_if_exception_type=(ValidationError,), stop_after_attempt=2)
                        )
        
        response: SampleAggregator = structured_llm.invoke(state.messages + [HumanMessage(self.prompts["aggregate_samples_prompt"])])
        largest_groups = response.get_largest_groups()
        messages = [
            AIMessage("========= Sample Aggregation Result ========="),
            AIMessage(str(largest_groups))
        ]
        return {"messages": messages, "largest_groups": largest_groups}
        # endregion

    def _resolution_router(self, state: AgentState) -> Literal["resolution", "choose_best_group"]:
        if len(state.largest_groups) == 1:
            return "resolution"
        else:
            return "choose_best_group"

    def _choose_best_group(self, state: AgentState) -> AgentState:
        class BestGroup(BaseModel):
            thought: str = Field("", description="The thought process of the model to choose the best group")
            best_group_index: int = Field(description="The index of the best group in the list of largest groups")

        best_group_selector = (
            RunnableLambda(
                lambda x: {
                    "groups": [
                        group_w_index
                        for i, group in enumerate(x["groups"])
                        for group_w_index in (
                            AIMessage(f"=========Group index: {i}=========="), 
                            AIMessage(group[0].answer)
                        )
                    ]
                } 
            )
            | ChatPromptTemplate.from_messages([
                ("placeholder", "{groups}"),
                ("user", 
                 self.prompts["best_group_prompt"].format(**{
                            "task": state.task, 
                            "long_term_goal": state.long_term_goal, 
                            "task_history": state.task_history
                        }
                    )
                )
            ])
            | self.llm.with_structured_output(BestGroup) 
        )

        response: BestGroup = best_group_selector.invoke({"groups": state.largest_groups})
        best_group = state.largest_groups[response.best_group_index]
        messages = [
            AIMessage("========= Best Group Selection ========="),
            AIMessage(response.thought),
            AIMessage(str(best_group))
        ]

        return {"messages": messages, "largest_groups": [best_group]}
    
    def _resolution(self, state: AgentState) -> AgentOutput:
        chain = (
            ChatPromptTemplate.from_messages([
                ("human", self.prompts["resolution_prompt"])
            ])
            | self.llm.with_structured_output(Solution)
        )
        solution = chain.invoke({"task": state.task, "context": state.largest_groups[0][0].answer})
        state.task.solution = solution
        return {"task": state.task}
    

# %% Testing the agent
graph = CoTSCAgent().create_graph()