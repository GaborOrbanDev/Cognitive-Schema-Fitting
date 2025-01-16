# %% Putting file to absolute path to be able to call schema packages
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#%% importing libraries
from langchain_community.callbacks import get_openai_callback
from schemas.agent_state_classes import AgentInput, Task
from schemas.cot_sc_schema import CoTSCAgent
from schemas.cot_schema import CoTAgent
from schemas.cot_w_sr_schema import CoTwSRAgent
from schemas.spp_schema import SPPAgent
from schemas.tot_schema import ToTAgent
import pandas as pd

#%% reading the data
data = pd.read_csv("./training/mmlu_all_w_id.csv", sep=";")

#%% defining the callback
agent = ToTAgent().create_agent()

with get_openai_callback() as cb:
    task_text="""Which one of the following is the most appropriate definition of a 99% confidence interval? [ "99% of the time in repeated samples, the interval would contain the true value of the parameter", "99% of the time in repeated samples, the interval would contain the estimated value of the parameter", "99% of the time in repeated samples, the null hypothesis will be rejected", "99% of the time in repeated samples, the null hypothesis will not be rejected when it was false" ]"""
    agent_input = AgentInput(task=Task(description=task_text))
    start_time = time.perf_counter()
    response = agent.invoke(agent_input)
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(response)
    print("====================================")
    print(f"Inference Time: {inference_time}")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Input Tokens: {cb.prompt_tokens}")
    print(f"Output Tokens: {cb.completion_tokens}")
    print(f"Cost: {cb.total_cost}")