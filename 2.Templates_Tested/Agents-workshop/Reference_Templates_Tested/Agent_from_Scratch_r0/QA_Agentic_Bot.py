from aisAgents import Agent
from aisTools import calculate
from ddg import ddg_search, ddg_news
from aisModels import GroqLLM
################################################WILL VARY#####################################################################################
# Define the Chain of Thought (COT) example
cot_example = """
Example session:

Question: Who is current president of USA?
Thought: I need to find the name of President of USA.
Action: ddg_search: 'who is President of USA'
PAUSE

You will be called again with this:

Observation: Donald Trump

If you have the answer, output it as the Answer.

Answer: The president of United State is Donald Trump
"""

###################################################################################################################################
tools={"calculate": calculate, "ddg_search": ddg_search, "ddg_news": ddg_news}
###################################################################################################################################
import os
os.system('clear')
# Initializing the LLM
llm = GroqLLM(api_key=os.environ.get("GROQ_API_KEY"))
# Creating an agent instance and pass the LLM instance to it
agent = Agent(llm=llm, cot_example=cot_example, tools=tools)
# Starting the agent's interaction loop
#agent.run(query='calculate the mass of two heaviest planets')
query = """Who is current president of Pakistan?"""
print(query)
print(100*"-")
agent.run(query=query)