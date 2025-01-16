"""
Tree of Thought cognitive schema implementation
source: https://arxiv.org/abs/2305.10601
"""

# %% Importing libraries
from __future__ import annotations
import os
from typing_extensions import Annotated
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSerializable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphRecursionError
import openai
from pydantic import BaseModel, Field, ValidationError
from schemas.agent_state_classes import AgentInput, AgentOutput, Solution

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# %% Schema state class
class Thought(BaseModel):
    """A `Thought` object represents a distinct thought within the cognition of the system"""

    content: AIMessage
    score: int | None = Field(
        default=None,
        description="The evaluation of the thought. It can be a number between 1 and 20, where 1 is the worst and 20 is the best."
    )
    context: list[AnyMessage] = []
    children: list[Thought] = []
    ancestors: list[Thought] = []


class AgentState(AgentInput, AgentOutput):
    messages: Annotated[list[AnyMessage], add_messages] = []
    steps: list[str] = []
    solution_candidates: list[Thought] = []


# %% Agent class
class ToTAgent:
    def __init__(self, prompt_file_path: str | None = None, prune_less_or_equal: int = 10) -> None:
        if prompt_file_path is None:
            prompt_file_path = "./prompts/tot_prompts.yaml"
        with open(prompt_file_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.cognition_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, top_p=0.7)
        self.evaluation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.prune_less_or_equal = prune_less_or_equal

    def create_agent(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState, input=AgentInput, output=AgentOutput)
        workflow.add_node("schema_setup", self._schema_setup)
        workflow.add_node("task_decomposition", self._task_decomposition)
        workflow.add_node("cognition", self._cognition)
        workflow.add_node("trace_optimatization", self._trace_optimatization)
        workflow.add_node("resolution", self._resolution)
        workflow.add_edge(START, "schema_setup")
        workflow.add_edge("schema_setup", "task_decomposition")
        workflow.add_edge("task_decomposition", "cognition")
        workflow.add_edge("cognition", "trace_optimatization")
        workflow.add_edge("trace_optimatization", "resolution")
        workflow.add_edge("resolution", END)
        return workflow.compile()

    def __call__(self) -> CompiledStateGraph:
        return self.create_agent()
    
    def __name__(self) -> str:
        return "ToT"
    # --------------------------------------------------------------------------------

    def _schema_setup(self, state: AgentInput) -> AgentState:
        prompt_template = SystemMessagePromptTemplate.from_template(self.prompts["system_prompt"])
        message = prompt_template.format(**{
            "long_term_goal": state.long_term_goal,
            "task_history": state.task_history,
            "task": state.task.description
        })
        return {"messages": [message]}
    
    def _task_decomposition(self, state: AgentState) -> AgentState:
        class TaskDecomposer(BaseModel):
            """Decompose the task into manageable subproblems"""
            analysis: str = Field(..., description="Analysis of the task")
            steps: list[str] = Field(..., description="Steps to solve the task")

        decomposer_llm = self.llm.with_structured_output(TaskDecomposer)
        response: TaskDecomposer = decomposer_llm.invoke(state.messages + [HumanMessage(self.prompts["task_decomposition_prompt"])])
        return {"messages": [AIMessage(f"{response.analysis}\nSteps: {response.steps}")], "steps": response.steps}
    
    def _cognition(self, state: AgentState) -> AgentState:
        # assigning chains
        cognition_chain = self._create_thought_generation_chain()
        evaluation_chain = self._create_evaluation_chain()

        # assigning variables for tree search
        parents: list[Thought] = [Thought(content=AIMessage("<SEED>"))]
        beam = 2

        for level, step in enumerate(state.steps):
            iter_count = 0

            # this is a placeholder wich will become parents
            children: list[Thought] = []

            # evaluation text is for cases when all children ware pruned and we need to retry
            # so it might contain contextual information for the next iteration
            evaluation_text: str | None = None

            while len(children) == 0 and iter_count < 3:
                iter_count += 1

                unevaled_children = cognition_chain.batch([
                    {
                        "long_term_goal": state.long_term_goal,
                        "task_history": state.task_history,
                        "task": state.task,
                        "step": step,
                        "context": parent.context + ([HumanMessage(evaluation_text)] if evaluation_text else []),
                        "ancestors": [*parent.ancestors, parent] if level > 0 else []
                    }
                    for parent in parents
                    for _ in range(beam)
                ])

                evaled_children, evaluation_text = evaluation_chain.invoke({
                    "long_term_goal": state.long_term_goal,
                    "task_history": state.task_history,
                    "task": state.task,
                    "step": state.steps,
                    "thoughts": unevaled_children
                }).values()

                children = list(filter(lambda x: x.score > self.prune_less_or_equal, evaled_children))

            # if still has has no children, raise an error
            if len(children) == 0:
                raise GraphRecursionError("Unable to create correct thought")
            # if children are found, assign them to parents
            else:
                parents = children

        return {"solution_candidates": parents}
    
    def _trace_optimatization(self, state: AgentState) -> AgentState:
        class Trace(BaseModel):
            scores: list[int] = []
            last_thought: Thought | None = None

            def avg_score(self) -> float:
                return sum([s * (0.1*(w+1)) for w, s in enumerate(self.scores)]) / sum([0.1*(w+1) for w in range(len(self.scores))])

            def get_context(self) -> list[AnyMessage]:
                return self.last_thought.context    
        
        traces: list[Trace] = []
        for thought in state.solution_candidates:
            scores = [a.score for a in thought.ancestors] + [thought.score]
            traces.append(Trace(scores=scores, last_thought=thought))

        for trace in traces:
            print(f"Scores: {trace.scores}, Avg: {trace.avg_score()}")
        best_trace = max(traces, key=lambda x: x.avg_score())

        return {"solution_candidates": [best_trace.last_thought]}
    
    def _resolution(self, state: AgentState) -> AgentOutput:
        chain = (
            ChatPromptTemplate.from_messages([
                state.messages[0],
                *state.solution_candidates[0].context,
                ("human", "{input}")
            ])
            | self.llm.with_structured_output(Solution)
        )
        solution = chain.invoke({"input": self.prompts["resolution_prompt"]})
        state.task.solution = solution
        return {"task": state.task}
        
    # --------------------------------------------------------------------------------
    # region chain creation methods             

    def _create_thought_generation_chain(self) -> RunnableSerializable[dict, Thought]:
        """Create the thought generation chain. 

        The chain creates the full chat context, samples the LLM and creates a `Thought` object."""

        cognition_chain = (            
            ChatPromptTemplate.from_messages([
                ("system", self.prompts["thought_generation_system_prompt"]),
                ("placeholder", "{context}"),
                ("user", self.prompts["thought_generation_prompt"])
            ])
            | RunnableParallel(
                response=self.cognition_llm,
                # get context cleared from system and user messages
                context=RunnableLambda(lambda prompt_template: prompt_template.messages[1:-1])
            )
            | RunnableParallel(
                response=lambda x: x["response"],
                context=lambda x: x["context"] + [x["response"]]
            )
        )

        chain = (
            RunnableParallel(
                cognition=cognition_chain,
                ancestors=lambda x: x["ancestors"]
            )
            | RunnableLambda(
                lambda x: Thought(
                    content=x["cognition"]["response"],
                    context=x["cognition"]["context"],
                    ancestors=x["ancestors"]
                )
            )
        )

        return chain
    
    def _create_evaluation_chain(self) -> RunnableSerializable[dict, dict[str, list[Thought] | str]]:
        """Create the evaluation chain.
        
        The chain evaluates the list of thoughts and returns the updated list of thoughts with scores and the text of the evaluation."""

        class EvalThoughts(BaseModel):
            """Evaluation structure for the thoughts"""
            evaluation_text: str = Field(..., description="Here you can think before answering")
            scores: list[int] = Field(..., description="List of scores. It can be a number between 1 and 20, where 1 is the worst and 20 is the best.")

        chain = (
            RunnableLambda(
                lambda x: {
                    **x,
                    "context": [
                        thought_w_index
                        for i, t in enumerate(x["thoughts"])
                        for thought_w_index in (
                            HumanMessage(f"=========Thought Candidate{i+1}=========="),
                            getattr(t, "content") 
                        )
                    ]
                } 
            )
            | (
                RunnableParallel(
                    evaluation=(
                        ChatPromptTemplate.from_messages(
                            [
                                ("system", self.prompts["evaluation_system_prompt"]),
                                ("placeholder", "{context}"),
                                ("user", self.prompts["evaluation_prompt"])
                            ]
                        ) 
                        | ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(EvalThoughts)
                    ),
                    thoughts=lambda x: x["thoughts"]
                )
                | RunnableParallel(
                    scores=lambda x: getattr(x["evaluation"], "scores"),
                    evaluation_text=lambda x: getattr(x["evaluation"], "evaluation_text"),
                    thoughts=lambda x: x["thoughts"]
                )
                | RunnableLambda(
                    lambda x: {
                        "thoughts": [
                            Thought(
                                content=getattr(x["thoughts"][i], "content"),
                                score=score,
                                context=getattr(x["thoughts"][i], "context"),
                                children=getattr(x["thoughts"][i], "children"),
                                ancestors=getattr(x["thoughts"][i], "ancestors")
                            )
                            for i, score in enumerate(x["scores"])
                        ],
                        "evaluation_text": x["evaluation_text"]
                    }
                )
            ).with_retry(retry_if_exception_type=(IndexError, ValidationError), stop_after_attempt=2)
        )

        return chain
    
    # endregion
    # --------------------------------------------------------------------------------


# %% Testing the agent
if __name__ == "__main__":
    graph = ToTAgent().create_agent()