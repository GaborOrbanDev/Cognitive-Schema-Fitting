# %% importing libraries
from pprint import pprint
import operator
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

# %% creating schema class

class Task(BaseModel):
    """`Task` object repesents a general but specific subproblem, decomoposed from the orginal `input_problem` of the user"""
    description: str = Field(..., description="Description with essential details of the task")
    

class SolvedTask(Task):
    """`SolvedTask` object represents a task that has been solved by the system"""
    solution: str = Field(..., description="The solution of the task")


class SchemaAgentInput(BaseModel):
    input_problem: str = Field(..., description="The original problem statement given by the user")
    task: Task = Field(..., description="The current task to be solved")
    task_history: list[SolvedTask] = Field([], description="List of tasks that have been solved so far")


class SchemaAgentOutput(BaseModel):
    solution: Any = Field("", description="The solution of the task")


class SampleResponse(BaseModel):
    """A sample response from the model for the given prompt"""

    chain_of_thought: Annotated[str, operator.add] = Field("", description="This is a scratchpad for the model to think on the question step by step")
    answer: Annotated[str, operator.add] = Field("", description="Here the model places the answer for the given question")


class SchemaAgentState(SchemaAgentInput, SchemaAgentOutput):
    messages: Annotated[list[AnyMessage], add_messages] = Field([], description="The inner messages of the agent")
    samples: list[SampleResponse] = []
    largest_groups: list[list[SampleResponse]] = []


# %% creating agent class

class SCAgent:
    def __init__(self) -> None:
        with open("./prompts/cot_sc_prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def schema_setup(self, state: SchemaAgentInput) -> SchemaAgentState:
        prompt_template = SystemMessagePromptTemplate.from_template(self.prompts["system_prompt"])
        message = prompt_template.format(**{
            "input_problem": state.input_problem,
            "task_history": state.task_history,
            "task": state.task.description
        })
        return {"messages": [message]}
    
    def sample_llm(self, state: SchemaAgentState) -> SchemaAgentState:
        sample_count = 10

        chain = (
            ChatPromptTemplate.from_messages([
                ("placeholder", "{context}"),
                ("user", "{input}")
            ])
            | RunnableLambda(
                lambda context: [context for _ in range(sample_count)]
            )
            | (self.llm.with_structured_output(SampleResponse)).map()
        )

        message = HumanMessagePromptTemplate.from_template(self.prompts["sample_llm_prompt"]).format(**{"task": state.task})
        samples = chain.invoke({"context": state.messages, "input": message.content})

        return {
            "messages": [message],
            "samples": samples
        }
    
    def aggregate_samples(self, state: SchemaAgentState) -> SchemaAgentState:

        class SampleGroup(BaseModel):
            sample_indexies: list[int] = Field([], description="List of samples that are similar to each other. Each item of is the sample index in the original list of samples")
        
        class SampleAggregator(BaseModel):
            scratchpad: str = Field("", description="Scratchpad for the model where it can think on the similarity of the different samples before grouping them")
            sample_groups: list[SampleGroup] = Field([], description="List of groups of samples that are similar to each other")
            
            def get_largest_groups(self) -> list[list[SampleResponse]]:
                length = max(len(group.sample_indexies) for group in self.sample_groups)
                return [[state.samples[i] for i in group.sample_indexies] 
                        for group in self.sample_groups 
                        if len(group.sample_indexies) == length]


        clustering_chain = (
            RunnableLambda(
                lambda x: {
                    "samples": [sample_w_index
                                for i, sample in enumerate(x["samples"])
                                for sample_w_index in (HumanMessage(f"=========Sample index: {i}=========="),
                                                        AIMessage(getattr(sample, "answer")))]
                } 
            )
            | ChatPromptTemplate.from_messages([
                ("placeholder", "{samples}"),
                ("user", self.prompts["aggregate_samples_prompt"])
            ])
            | RunnableParallel(
                response=self.llm.with_structured_output(SampleAggregator).with_retry(retry_if_exception_type=(ValidationError,), stop_after_attempt=2),
                context=RunnablePassthrough()
            )
            | RunnableLambda(
                lambda x: {
                    "largest_groups": x["response"].get_largest_groups(),
                    "context": getattr(x["context"], "messages") + [AIMessage(content=str(x["response"]))]
                }
            )
        )

        largest_groups, context = clustering_chain.invoke({"samples": state.samples}).values()

        return {
            "messages": context,
            "largest_groups": largest_groups
        }
    
    def choose_best_group(self, state: SchemaAgentState) -> SchemaAgentState:
        class BestGroup(BaseModel):
            thought: str = Field("", description="The thought process of the model to choose the best group")
            best_group_index: int = Field(description="The index of the best group in the list of largest groups")

        summarize_chain = (
            RunnableLambda(
                lambda x: {"group": [("assistant", sample.answer) for sample in x["group"]]}
            )
            | ChatPromptTemplate.from_messages([
                state.messages[0],
                ("placeholder", "{group}"),
                ("user", self.prompts["summarize_prompt"])
            ])
            | self.llm
        )

        best_group_selector = (
            RunnableLambda(
                lambda x: {
                    "summaries": [summary_w_index
                                for i, summary in enumerate(x["summaries"])
                                for summary_w_index in (HumanMessage(f"=========Summry index: {i}=========="), summary)]
                } 
            )
            | ChatPromptTemplate.from_messages([
                state.messages[0],
                ("placeholder", "{summaries}"),
                ("user", self.prompts["best_group_prompt"])
            ])
            | self.llm.with_structured_output(BestGroup) 
        )

        summaries = summarize_chain.batch([{"group": group} for group in state.largest_groups])

        if len(summaries) == 1:
            return {
                "messages": summaries,
                "solution": summaries[0].content
            }
        else:
            best_group = best_group_selector.invoke({"summaries": summaries})
            return {
                "messages": summaries + [AIMessage(str(best_group)), summaries[best_group.best_group_index]],
                "solution": summaries[best_group.best_group_index].content
            }
    
    # ---------------------------------------------------------------------------
    
    def __call__(self) -> CompiledStateGraph:
        workflow = StateGraph(SchemaAgentState, input=SchemaAgentInput, output=SchemaAgentOutput)
        workflow.add_node("setup", self.schema_setup)
        workflow.add_node("sample_llm", self.sample_llm)
        workflow.add_node("aggregate_samples", self.aggregate_samples)
        workflow.add_node("choose_best_group", self.choose_best_group)

        workflow.add_edge(START, "setup")
        workflow.add_edge("setup", "sample_llm")
        workflow.add_edge("sample_llm", "aggregate_samples")
        workflow.add_edge("aggregate_samples", "choose_best_group")
        workflow.add_edge("choose_best_group", END)
        return workflow.compile()
    
    def create_graph(self):
        return self.__call__()

        
    

# %% creating graph

graph = SCAgent().create_graph()

# %%
# prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, logprobs=True)
# chain = prompt | llm
# pprint(chain.invoke({"input": "What is the capital of France?"})
# )
# %%
