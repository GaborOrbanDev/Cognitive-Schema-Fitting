"""
Multi-Persona Self-Collaboration - Solo Performance Promping cognitive schema implementation
source: https://arxiv.org/abs/2307.05300
"""

# %% Importing libraries
import os
from typing import Annotated
from typing_extensions import Literal
import openai
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import RetryPolicy
from pydantic import BaseModel, Field
from pydantic_core import ValidationError
from schemas.agent_state_classes import AgentInput, AgentOutput, Solution

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# %% Schema state class
class AgentState(AgentInput, AgentOutput):
    messages: Annotated[list[AnyMessage], add_messages] = []


# %% Agent class
class SPPAgent:
    def __init__(self, temperature: float = 0.5, prompt_file_path: str | None = None, max_eval_count: int = 3) -> None:
        if prompt_file_path is None:
            prompt_file_path = "./prompts/spp_prompts.yaml"
        with open(prompt_file_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        self.max_eval_count = max_eval_count

    def create_agent(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState, input=AgentInput, output=AgentOutput)
        workflow.add_node("schema_setup", self._schema_setup)
        workflow.add_node("persona_identification", self._persona_identification)
        workflow.add_node("brainstorming", self._brainstorming)
        workflow.add_node("drafter", self._drafter)
        workflow.add_node("feedback", self._feedback, retry=RetryPolicy(retry_on=(ValidationError,)))
        workflow.add_node("resolution", self._resolution)
        workflow.add_edge(START, "schema_setup")
        workflow.add_edge("schema_setup", "persona_identification")
        workflow.add_edge("persona_identification", "brainstorming")
        workflow.add_edge("brainstorming", "drafter")
        workflow.add_edge("drafter", "feedback")
        workflow.add_conditional_edges("feedback", self._refinement_router)
        workflow.add_edge("resolution", END)
        return workflow.compile()

    def __call__(self):
        return self.create_agent()
    
    def __name__(self) -> str:
        return "SPP"
    
    # --------------------------------------------------------------------------------

    def _schema_setup(self, state: AgentInput) -> AgentState:
        prompt_template = SystemMessagePromptTemplate.from_template(self.prompts["system_prompt"])
        message = prompt_template.format(**{
            "long_term_goal": state.long_term_goal,
            "task_history": state.task_history,
            "task": state.task.description
        })
        return {"messages": [message]}
    
    def _get_prefix(self, caller: str) -> AIMessage:
        return AIMessage(f"============= {caller.capitalize().replace('_', ' ')} =============\n", section=caller)
    
    def _persona_identification(self, state: AgentState) -> AgentState:
        context = state.messages + [HumanMessage(self.prompts["persona_identification_prompt"])]
        prefix = self._get_prefix("persona_identification")
        return {"messages": [prefix, self.llm.invoke(context)]}
    
    def _brainstorming(self, state: AgentState) -> AgentState:
        context = state.messages + [HumanMessage(self.prompts["brainstorming_prompt"])]
        prefix = self._get_prefix("brainstorming")
        return {"messages": [prefix, self.llm.invoke(context)]}
    
    def _drafter(self, state: AgentState) -> AgentState:
        solution_i = len([m for m in state.messages if getattr(m, "section", None) == "draft"])
        prompt = self.prompts["first_round_draft_prompt"] if solution_i == 0 else self.prompts["later_rounds_draft_prompt"]
        context = state.messages + [HumanMessage(prompt)]
        prefix = self._get_prefix("draft")
        return {"messages": [prefix, self.llm.invoke(context)]}
    
    def _feedback(self, state: AgentState) -> AgentState:
        class PersonaFeedback(BaseModel):
            """Structure for persona feedback"""
            name: str = Field(..., description="Name of the persona")
            feedback: str = Field(..., description="Feedback from the perspective of the persona")

        class Feedback(BaseModel):
            """Structure for feedback"""
            feedbacks_from_personas: list[PersonaFeedback] = Field(..., description="Feedbacks from different personas")
            ai_assistant_decision: Literal["resolution", "drafter"] = Field(..., description="Decision made by the AI assistant. Resolution if the task is ready for resolution, drafter if the task needs more refinement.")

        # creating structured LLM
        structured_llm = self.llm.with_structured_output(Feedback)
        # getting the number of previous feedbacks
        feedback_i = len([m for m in state.messages if getattr(m, "section", None) == "later_rounds_feedback_prompt"])
        # assigning the prompt
        prompt = self.prompts["first_round_feedback_prompt"] if feedback_i == 0 else self.prompts["later_rounds_feedback_prompt"]

        feedback: Feedback = structured_llm.invoke(state.messages + [HumanMessage(prompt)])
        prefix = self._get_prefix("feedback")
        return {"messages": [prefix, AIMessage(str(feedback), decision=feedback.ai_assistant_decision)]}
    
    def _refinement_router(self, state: AgentState) -> Literal["resolution", "drafter"]:
        eval_count = len([m for m in state.messages if getattr(m, "section", None) == "feedback"])
        decision: str | None = getattr(state.messages[-1], "decision", None)       
        if eval_count >= self.max_eval_count:
            return "resolution"
        elif isinstance(decision, str) and decision in ["resolution", "drafter"]:
            return decision
        else:
            return "drafter"
        
    def _resolution(self, state: AgentState) -> AgentOutput:
        chain = (
            ChatPromptTemplate.from_messages([
                state.messages[0], # system prompt
                state.messages[-3], # last proposal
                state.messages[-1], # feedback
                ("human", "{input}")
            ])
            | self.llm.with_structured_output(Solution)
        )
        solution = chain.invoke({"input": self.prompts["resolution_prompt"]})
        state.task.solution = solution
        return {"task": state.task}
    

# %% Testing the agent
if __name__ == "__main__":
    graph = SPPAgent().create_agent()    