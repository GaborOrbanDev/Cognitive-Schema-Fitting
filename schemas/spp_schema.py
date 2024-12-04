# %% importing libraries
import os
from typing import Annotated
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel, Field


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
    solution: str = Field("", description="The solution of the task")


class SchemaAgentState(SchemaAgentInput, SchemaAgentOutput):
    messages: Annotated[list[AnyMessage], add_messages] = Field([], description="The inner messages of the agent")


# %% creating agent class

class SPPAgent:
    def __init__(self) -> None:
        with open("./prompts/spp_prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)

        self.llm = ChatOpenAI(model="gpt-4o-mini")

        self.node_context_returner_llm_chain: RunnableSerializable[dict, list[AnyMessage]] = (
            ChatPromptTemplate.from_messages([
                ("placeholder", "{context}"),
                ("user", "{prompt}")
            ])
            | RunnableParallel(
                response=self.llm,
                context=RunnablePassthrough()
            )
            | RunnableLambda(
                lambda x: [getattr(x["context"], "messages")[-1]] + [x["response"]]
            )
        )

    def schema_setup(self, state: SchemaAgentInput) -> SchemaAgentState:
        system_prompt = SystemMessage(
            self.prompts["system_prompt"].format(
                **{
                    "input_problem": state.input_problem,
                    "task_history": state.task_history,
                    "task": state.task
                }
            )
        )
        
        return {
            "messages": [system_prompt]
        }
    
    def persona_indentification(self, state: SchemaAgentState) -> SchemaAgentState:
        response_context = self.node_context_returner_llm_chain.invoke({
                "context": state.messages,
                "prompt": self.prompts["persona_identification_prompt"]
        })
        return {
            "messages": response_context
        }
    
    def brainstorming(self, state: SchemaAgentState) -> SchemaAgentState:
        response_context = self.node_context_returner_llm_chain.invoke({
                "context": state.messages,
                "prompt": self.prompts["brainstorming_prompt"]
        })
        return {
            "messages": response_context
        }
    
    def initial_solution_proposal(self, state: SchemaAgentState) -> SchemaAgentState:
        if len([message for message in state.messages if getattr(message, "source_node", None) == "initial_solution_proposal"]) == 0:
            prompt = self.prompts["first_round_initial_solution_proposal_prompt"]
        else:
            prompt = self.prompts["later_rounds_solution_proposal_prompt"]
        
        response_context = self.node_context_returner_llm_chain.invoke({
                "context": state.messages,
                "prompt": prompt
        })

        setattr(response_context[-1], "source_node", "initial_solution_proposal")

        return {
            "messages": response_context
        }
    
    def feedback(self, state: SchemaAgentState) -> SchemaAgentState:
        if len([message for message in state.messages if getattr(message, "source_node", None) == "feedback"]) == 0:
            prompt = self.prompts["first_round_feedback_prompt"]
        else:
            prompt = self.prompts["later_rounds_feedback_prompt"]

        response_context = self.node_context_returner_llm_chain.invoke({
                "context": state.messages,
                "prompt": prompt
        })

        setattr(response_context[-1], "source_node", "feedback")

        return {
            "messages": response_context
        }
    
    def revision_router(self, state: SchemaAgentState) -> str:
        number_of_revisions = len([message for message in state.messages if getattr(message, "source_node", None) == "feedback"])
        last_message = state.messages[-1]
        if "<OK>" in last_message.content or number_of_revisions > 5:
            return "resolve"
        else:
            return "revise"



        



# %%
agent = SPPAgent()
workflow = StateGraph(SchemaAgentState, input=SchemaAgentInput)
workflow.add_node("schema_setup", agent.schema_setup)
workflow.add_node("persona_indentification", agent.persona_indentification)
workflow.add_node("brainstorming", agent.brainstorming)
workflow.add_node("initial_solution_proposal", agent.initial_solution_proposal)
workflow.add_node("feedback", agent.feedback)

workflow.add_edge(START, "schema_setup")
workflow.add_edge("schema_setup", "persona_indentification")
workflow.add_edge("persona_indentification", "brainstorming")
workflow.add_edge("brainstorming", "initial_solution_proposal")
workflow.add_edge("initial_solution_proposal", "feedback")
workflow.add_conditional_edges(
    "feedback",
    agent.revision_router,
    {
        "resolve": END,
        "revise": "initial_solution_proposal"
    }
)

graph = workflow.compile()