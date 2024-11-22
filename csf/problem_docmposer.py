import os
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .definitions import AgentMessagesState, ProblemDecompositionInput, ProblemDecompositionOutput


class DecomposerNodes:
    def __init__(self) -> None:
        with open("prompts/decomposition_prompts.yaml", "r") as file:
            self.prompts = yaml.safe_load(file)

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
        self.eval_llm = ChatOpenAI(model="gpt-4o", temperature=0.7, stop_sequences=["<</STOP>>"])
        self.response_llm = (ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
                             .with_structured_output(ProblemDecompositionOutput))
        
    def intitial_problem_decomposition(self, state: ProblemDecompositionInput) -> AgentMessagesState:        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts["system_prompt"]),
            ("human", self.prompts["initial_pd_suffix_instruction"])
        ])

        chain = prompt | self.llm
        
        return {"messages": [*prompt.invoke(state.input_problem).to_messages(), 
                             chain.invoke(state.input_problem)]}
    
    def self_evaluate(self, state: AgentMessagesState) -> AgentMessagesState:
        prompt = ChatPromptTemplate.from_messages([
            *state.messages,
            ("human", "{input}")
        ])

        chain = prompt | self.eval_llm

        context = [HumanMessage(self.prompts["evaluation_prompt"]), chain.invoke(self.prompts["evaluation_prompt"])]

        if context[-1].content.endswith("<OK>"):
            return {"messages": context, "is_ok": True}
        else:
            return {"messages": context, "is_ok": False}
        

if __name__ == "__main__":
    load_dotenv()