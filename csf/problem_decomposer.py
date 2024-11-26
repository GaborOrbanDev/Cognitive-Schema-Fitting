from __future__ import annotations

import os
from typing import Annotated
import yaml

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.graph.state import CompiledStateGraph
import openai
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------------
# region Task definitions

class Task(BaseModel):
    """`Task` object repesents a general but specific subproblem, decomoposed from the orginal `input_problem` of the user"""
    description: str = Field(..., description="Description with essential details of the task")
    

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


class AgentMessagesState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field([])

class ProblemDecompositionState(
        ProblemDecompositionInput, 
        ProblemDecompositionOutput,
        AgentMessagesState
    ):
    pass

# endregion
# ---------------------------------------------------------------------------------
# region Definitions for cognitive schema selection


class ProblemDecomposer:
    def __init__(self) -> None:
        with open("./prompts/iterative_decomposition.yaml", "r") as file:
            self.prompts = yaml.safe_load(file)

        self.llm = ChatOpenAI(model="gpt-4o")
        self.response_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    
    def create_graph(self) -> CompiledStateGraph:
        return self.__call__()

    def __call__(self) -> CompiledStateGraph:
        workflow = StateGraph(ProblemDecompositionState, input=ProblemDecompositionInput, output=ProblemDecompositionOutput)

        workflow.add_node("decompose", self.decompose)
        workflow.add_node("verify", self.verify)
        workflow.add_node("resolve", self.resolve)

        workflow.add_edge(START, "decompose")
        workflow.add_edge("decompose", "verify")
        workflow.add_conditional_edges(
            "verify",
            lambda state: getattr(state.messages[-1], "is_verified", False) or (len(state.messages) > 7),
            {
                False: "verify",
                True: "resolve"
            }
        )
        workflow.add_edge("resolve", END)

        return workflow.compile()

    def decompose(self, state: ProblemDecompositionInput) -> ProblemDecompositionState:
        system_prompt = (SystemMessagePromptTemplate
                            .from_template(self.prompts["system_prompt"]))
        prompt = ChatPromptTemplate.from_messages([system_prompt])
        chain = prompt | RunnableParallel(
            response = self.llm,
            context = RunnablePassthrough()
        ) | RunnableLambda(lambda x: x["context"].messages + [x["response"]])
        context = chain.invoke({"input_problem": state.input_problem})
        return {"messages": context}


    def verify(self, state: ProblemDecompositionState) -> ProblemDecompositionState:
        prompt = ChatPromptTemplate.from_messages([
            ("placeholder", "{context}"),
            ("human", "{verifier_prompt}")
        ])
        chain = (
            prompt 
            | RunnableParallel(
                response = self.llm,
                context = RunnableLambda(lambda x: [x.messages[-1]])
            ) 
            | RunnableLambda(
                lambda x: {
                    "response": x["response"],
                    "context": x["context"] + [x["response"]]
                }
            )
        )
        response, context = chain.invoke({"context": state.messages, "verifier_prompt": self.prompts["verifier_prompt"]}).values()
        if "<!OK>" in response.content:
            setattr(context[-1], "is_verified", True)
        else:
            setattr(context[-1], "is_verified", False)
        return {"messages": context}


    def resolve(self, state: ProblemDecompositionState) -> ProblemDecompositionOutput:
        prompt = ChatPromptTemplate.from_messages([
            ("placeholder", "{context}"),
            ("user", "{input}")
        ])
        llm = (self.response_llm
            .with_structured_output(ProblemDecompositionOutput)
            .with_retry(retry_if_exception_type=(ValidationError,), stop_after_attempt=2))
        chain = prompt | llm
        response = chain.invoke({"context": state.messages[-2:], "input": self.prompts["resolution_prompt"]}) 
        return response


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    graph = ProblemDecomposer().create_graph()