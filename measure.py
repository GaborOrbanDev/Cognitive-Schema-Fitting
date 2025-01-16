#%% importing libraries
from langchain_community.callbacks import get_openai_callback
from schemas.tot_schema import ToTAgent, AgentInput, Task

#%% defining the callback
agent = ToTAgent().create_graph()

with get_openai_callback() as cb:
    task_text="""Which one of the following is the most appropriate definition of a 99% confidence interval? [ "99% of the time in repeated samples, the interval would contain the true value of the parameter", "99% of the time in repeated samples, the interval would contain the estimated value of the parameter", "99% of the time in repeated samples, the null hypothesis will be rejected", "99% of the time in repeated samples, the null hypothesis will not be rejected when it was false" ]"""
    agent_input = AgentInput(task=Task(description=task_text))
    response = agent.invoke(agent_input)
    print(response)
    print("===================================")
    print(f"Total Tokens: {cb.total_tokens}")