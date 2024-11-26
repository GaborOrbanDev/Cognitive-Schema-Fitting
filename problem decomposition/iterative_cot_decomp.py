# %%
import os
from typing import Annotated
import yaml

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnablePick, RunnableSequence
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import openai
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# %%
class Task(BaseModel):
    description: str = Field(..., title="Description", description="The task description")

class DecompInput(BaseModel):
    input_problem: str = Field(..., title="Input Problem", description="The input problem to be decomposed")

class DecompOutput(BaseModel):
    tasks: list[Task] = Field([], title="Tasks", description="The tasks to be done to solve the input problem")

class DecompGraphState(DecompInput, DecompOutput):
    messages: Annotated[list[AnyMessage], add_messages] = Field([], title="Messages", description="The messages in the graph")

# %%
with open("./prompts/iterative_decomposition.yaml") as f:
    PROMPTS = yaml.safe_load(f)

# %%
def decompose(state: DecompInput) -> DecompGraphState:
    system_prompt = (SystemMessagePromptTemplate
                        .from_template(PROMPTS["system_prompt"]))
    prompt = ChatPromptTemplate.from_messages([system_prompt])
    chain = prompt | RunnableParallel(
        response = ChatOpenAI(model="gpt-4o"),
        context = RunnablePassthrough()
    ) | RunnableLambda(lambda x: x["context"].messages + [x["response"]])
    context = chain.invoke({"input_problem": state.input_problem})
    return {"messages": context}


def verify(state: DecompGraphState) -> DecompGraphState:
    prompt = ChatPromptTemplate.from_messages([
        ("placeholder", "{context}"),
        ("human", "{verifier_prompt}")
    ])
    llm = ChatOpenAI(model="gpt-4o")
    chain = (
        prompt 
        | RunnableParallel(
            response = llm,
            context = RunnableLambda(lambda x: [x.messages[-1]])
        ) 
        | RunnableLambda(
            lambda x: {
                "response": x["response"],
                "context": x["context"] + [x["response"]]
            }
        )
    )
    response, context = chain.invoke({"context": state.messages, "verifier_prompt": PROMPTS["verifier_prompt"]}).values()
    if "<!OK>" in response.content:
        setattr(context[-1], "is_verified", True)
    else:
        setattr(context[-1], "is_verified", False)
    return {"messages": context}


def resolve(state: DecompGraphState) -> DecompOutput:
    prompt = ChatPromptTemplate.from_messages([
        ("placeholder", "{context}"),
        ("user", "{input}")
    ])
    llm = (ChatOpenAI(model="gpt-4o-mini", temperature=0)
           .with_structured_output(DecompOutput)
           .with_retry(retry_if_exception_type=(ValidationError,), stop_after_attempt=2))
    chain = prompt | llm
    response = chain.invoke({"context": state.messages[-2:], "input": PROMPTS["resolution_prompt"]}) 
    return response

# %%
workflow = StateGraph(DecompGraphState, input=DecompInput, output=DecompGraphState)
workflow.add_node("decompose", decompose)
workflow.add_node("verify", verify)
workflow.add_node("resolve", resolve)
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
graph = workflow.compile()