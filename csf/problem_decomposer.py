from __future__ import annotations

import os
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
import openai
from pprint import pprint
from definitions import AgentMessagesState, ProblemDecompositionInput, ProblemDecompositionOutput


class ProblemDecomposer:
    def __init__(self) -> None:
        with open("prompts/decomposition_prompts.yaml", "r") as file:
            self.prompts = yaml.safe_load(file)

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
        self.eval_llm = ChatOpenAI(model="gpt-4o", temperature=0.7, stop_sequences=["<</STOP>>"])
        self.response_llm = (
            ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            .with_structured_output(ProblemDecompositionOutput)
        )

    def __call__(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentMessagesState, input=ProblemDecompositionInput, output=ProblemDecompositionOutput)

        workflow.add_node("initial_problem_decomposition", self.intitial_problem_decomposition)
        workflow.add_node("self_evaluate", self.self_evaluate)
        workflow.add_node("self_refine", self.self_refine)
        workflow.add_node("resolution", self.resolution)

        workflow.add_edge(START, "initial_problem_decomposition")
        workflow.add_edge("initial_problem_decomposition", "self_evaluate")
        workflow.add_conditional_edges(
            "self_evaluate",
            lambda state: getattr(state.messages[-1], "is_evaluation_ok", False),
            {
                True: "resolution",
                False: "self_refine"
            }
        )
        workflow.add_edge("self_refine", "self_evaluate")
        workflow.add_edge("resolution", END)

        return workflow.compile()

        
    def intitial_problem_decomposition(self, state: ProblemDecompositionInput) -> AgentMessagesState:        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts["system_prompt"]),
            ("human", self.prompts["initial_pd_suffix_instruction"])
        ])
        chain = prompt | self.llm
        return {"messages": prompt.invoke(state.input_problem).to_messages() + [chain.invoke(state.input_problem)]}
    
    def self_evaluate(self, state: AgentMessagesState) -> AgentMessagesState:
        prompt = ChatPromptTemplate.from_messages([*state.messages, ("human", "{input}")])
        chain = prompt | self.eval_llm
        evaluation_context: list[AnyMessage] = [HumanMessage(self.prompts["evaluation_prompt"]), chain.invoke(self.prompts["evaluation_prompt"])]
        if evaluation_context[-1].content.endswith("<OK>"):
            setattr(evaluation_context[-1], "is_evaluation_ok", True)
        else:
            setattr(evaluation_context[-1], "is_evaluation_ok", False)
        return {"messages": evaluation_context}
    
    def self_refine(self, state: AgentMessagesState) -> AgentMessagesState:
        prompt = ChatPromptTemplate.from_messages([*state.messages, ("human", "{input}")])
        chain = prompt | self.llm
        return {"messages": [HumanMessage(self.prompts["refinement_prompt"]), chain.invoke(self.prompts["refinement_prompt"])]}

    def resolution(self, state: AgentMessagesState) -> ProblemDecompositionOutput:
        prompt = ChatPromptTemplate.from_messages([*state.messages, ("human", "{input}")])
        chain = prompt | self.response_llm
        return chain.invoke(self.prompts["resolution_prompt"])  
        

if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    problem_decomposer = ProblemDecomposer()
    graph = problem_decomposer()

    context = graph.stream(ProblemDecompositionInput(input_problem="I want to build a house."))
    for state in context:
        pprint(state)
        print("\n")
        print("---------------------------------")
        print("\n")
