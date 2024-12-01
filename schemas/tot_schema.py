# %%
from __future__ import annotations

import os
from operator import add
from typing_extensions import TypedDict, Annotated
import yaml
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnablePick, RunnableSequence
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.errors import GraphRecursionError
import openai
from pprint import pprint
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# %%
class Task(BaseModel):
    """`Task` object repesents a general but specific subproblem, decomoposed from the orginal `input_problem` of the user"""
    description: str = Field(..., description="Description with essential details of the task")
    

class SolvedTask(Task):
    """`SolvedTask` object represents a task that has been solved by the system"""
    solution: str = Field(..., description="The solution of the task")


class Thought(BaseModel):
    """A `Thought` object represents a distinct thought within the cognition of the system"""

    thought: AIMessage
    evaluation: float | None = Field(
        default=None,
        description="The evaluation of the thought. It can be a number between 0 and 1.0 being 0 the worst and 1.0 the best."
    )
    context: list[AnyMessage] = []
    children: list[Thought] = []


class SchemaAgentInput(BaseModel):
    input_problem: str = Field(..., description="The original problem statement given by the user")
    task: Task = Field(..., description="The current task to be solved")
    task_history: list[SolvedTask] = Field([], description="List of tasks that have been solved so far")


class SchemaAgentOutput(BaseModel):
    solution: str = Field("", description="The solution of the task")


class ThoughtProcess(SchemaAgentInput, SchemaAgentOutput):
    """A `ThoughtProcess` object represents the thought process of the system"""

    steps: list[str] = []
    thought: Thought | None = None


# %%
class ToTAgent:
    def __init__(self) -> None:
        with open("./prompts/tot_prompts.yaml") as f:
            self.prompts = yaml.safe_load(f)

        self.decomp_llm = ChatOpenAI(model="gpt-4o")

        # high temperature for more creative responses, low top_p for more likely responses
        # source: https://medium.com/@1511425435311/understanding-openais-temperature-and-top-p-parameters-in-language-models-d2066504684f
        self.cognition_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, top_p=0.7)
        self.evaluation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.resolution_llm = ChatOpenAI(model="gpt-4o")

    # --------------------------------------------------------------------------------

    def schema_setup(self, state: SchemaAgentInput) -> ThoughtProcess:
        class Steps(BaseModel):
            analysis: str = Field(..., description="Analysis of the task")
            steps: list[str]

        prompt = ChatPromptTemplate.from_messages(
                    [("system", self.prompts["system_prompt"]), 
                    ("user", self.prompts["task_decomposition_prompt"])]
                )
        llm = self.decomp_llm.with_structured_output(Steps)
        chain = prompt | llm | RunnableLambda(lambda x: getattr(x, "steps"))

        steps: list[str] = chain.invoke({
            "input_problem": state.input_problem,
            "task_history": state.task_history,
            "task": state.task
        })

        return {"steps": steps}
    
    # --------------------------------------------------------------------------------

    def cognition(self, state: ThoughtProcess) -> ThoughtProcess:
        tree = Thought(thought=AIMessage("<SEED>"))
    
        self._bfs_cognition_walk(state=state, parents=[tree], level=0)

        return {"thought": tree}


    def _bfs_cognition_walk(self, state: ThoughtProcess, parents: list[Thought], level: int, beam: int = 2):
        """Breath First Search to walk through the cognition tree

        This approach is much faster then the Depth First Search because of batch processing.

        Args:
            state (ThoughtProcess): state of the graph
            parents (list[Thought]): list of parent thoughts
            level (int): current level of the tree
            beam (int, optional): number of alternative thoughts on the same branch. Defaults to 2.

        Returns:
            None
                It modifies the parents list in place.

        Raises:
            GraphRecursionError: if the system is unable to create a correct thought
        """

        cognition_chain = self._create_thought_generation_chain()
        evaluation_chain = self._create_evaluation_chain()

        children: list[Thought] = []
        iter_count: int = 0
        evaluation_text: str | None = None

        while iter_count < 3:
            iter_count += 1

            raw_children = cognition_chain.batch([
                {
                    "input_problem": state.input_problem,
                    "task": state.task,
                    "step": state.steps[level],
                    "context": parent.context + ([HumanMessage(evaluation_text)] if evaluation_text else [])
                }
                for parent in parents
                for _ in range(beam)
            ])

            evaled_children, evaluation_text = evaluation_chain.invoke({
                "input_problem": state.input_problem,
                "task": state.task,
                "step": state.steps,
                "thoughts": raw_children
            }).values()

            children.extend(list(filter(lambda x: x.evaluation >= 0.5, evaled_children)))

            if len(children) > 0:
                break

        if len(children) == 0:
            raise GraphRecursionError("Unable to create correct thought")

        # if walk has not reached the end of the tree, continue walking
        if level < len(state.steps) - 1:
            self._bfs_cognition_walk(state, children, level + 1)

        # if walk has aleady reached the end of the tree, and returned to the root
        if level == 0:
            parents[0].children.extend(children)
        # if walk has reached the end of the tree, and returning to the parent
        else:
            for child in children:
                for parent in parents:
                    if parent.context[-1].content == child.context[-3].content:
                        parent.children.append(child)
                        break
        
        return None


    def _create_thought_generation_chain(self):
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.prompts["thought_generation_system_prompt"]),
                ("placeholder", "{context}"),
                ("user", self.prompts["thought_generation_prompt"])
            ])
            | RunnableParallel(
                response=self.cognition_llm,
                context=RunnableLambda(lambda p: p.messages[1:])
            )
            | RunnableParallel(
                response=RunnablePick(keys="response"),
                context=lambda x: x["context"] + [x["response"]]
            ) 
            | RunnableLambda(
                lambda x: Thought(
                    thought=x["response"],
                    context=x["context"]
                )
            )
        )
        return chain
    

    def _create_evaluation_chain(self):
        class EvalResults(BaseModel):
            evaluation_text: str = Field(..., description="Here you can think before answering")
            scores: list[float] = Field(..., description="List of scores [0-1], in the order of thoughts within the context")

        chain = (
            RunnableLambda(
                lambda x: {
                    **x,
                    "context": [thought_w_index
                                for i, t in enumerate(x["thoughts"])
                                for thought_w_index in (HumanMessage(f"=========Thought Candidate{i+1}=========="),
                                                        getattr(t, "thought") )]
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
                        | ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(EvalResults)
                    ),
                    thoughts=lambda x: x["thoughts"]
                )
                | RunnableParallel(
                    evaluation=lambda x: getattr(x["evaluation"], "scores"),
                    evaluation_text=lambda x: getattr(x["evaluation"], "evaluation_text"),
                    thoughts=lambda x: x["thoughts"]
                )
                | RunnableLambda(
                    lambda x: {
                        "thoughts": [
                            Thought(
                                thought=getattr(x["thoughts"][i], "thought"),
                                evaluation=score,
                                context=getattr(x["thoughts"][i], "context"),
                                children=getattr(x["thoughts"][i], "children")
                            )
                            for i, score in enumerate(x["evaluation"])
                        ],
                        "evaluation_text": x["evaluation_text"]
                    }
                )
            ).with_retry(retry_if_exception_type=(IndexError, ValidationError), stop_after_attempt=2)
        )

        return chain

    # --------------------------------------------------------------------------------

    def trace_optimization(self, state: ThoughtProcess) -> SchemaAgentOutput:
        class Trace(BaseModel):
            scores: list[float] = []
            final_thought: Thought | None = None

            def avg_score(self) -> float:
                return sum(self.scores) / len(self.scores)
            
            def get_full_context(self) -> list[AnyMessage]:
                return self.final_thought.context + [self.final_thought.thought]
            

        traces: list[Trace] = []

        def trace_walk(thought: Thought, trace: Trace):
            for child_thought in thought.children:
                trace_i = trace.model_copy(deep=True)
                trace_i.scores.append(child_thought.evaluation)
                if child_thought.children:
                    trace_walk(child_thought, trace_i)
                else:
                    trace_i.final_thought = child_thought
                    traces.append(trace_i)

        trace_walk(state.thought, Trace())
        
        traces = list(filter(lambda x: len(x.scores) == len(state.steps), traces))

        max_trace_value = max([trace.avg_score() for trace in traces])
        for trace in traces:
            if trace.avg_score() == max_trace_value:
                best_trace = trace
                break
        
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("placeholder", "{context}"),
                    ("user", self.prompts["best_trace_prompt"])
                ]
            )
            | self.resolution_llm
            | StrOutputParser()
        )

        solution = chain.invoke({
            "context": best_trace.get_full_context(),
            "task": state.task
        })

        return {"solution": solution}

        



# %%
agent = ToTAgent()

workflow = StateGraph(state_schema=ThoughtProcess, input=SchemaAgentInput, output=SchemaAgentOutput)
workflow.add_node("setup", agent.schema_setup)
workflow.add_node("cognition", agent.cognition)
workflow.add_node("trace_optimization", agent.trace_optimization)
workflow.add_edge(START, "setup")
workflow.add_edge("setup", "cognition")
workflow.add_edge("cognition", "trace_optimization")
workflow.add_edge("trace_optimization", END)

graph = workflow.compile()

# %%
# response = graph.invoke(SchemaAgentInput(
#     input_problem="is tomato puree and tomato sauce the same thing?",
#     task=Task(description="Find the difference between tomato puree and tomato sauce"),
#     task_history=[]
# ))

# # %%
# pprint(response)

# # %%
# with open("../prompts/tot_prompts.yaml") as f:
#     PROMPTS = yaml.safe_load(f)

