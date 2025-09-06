from aisAgents import Agent
from aisTools import calculate, get_planet_mass, currency_converter
from ddg import ddg_search, ddg_news
from aisModels import GroqLLM
################################################WILL VARY#####################################################################################
# Define the Chain of Thought (COT) example
cot_example = """
Example session:

Question: What is the mass of Earth times 2?
Thought: I need to find the mass of Earth.
Action: get_planet_mass: Earth
PAUSE

You will be called again with this:

Observation: 5.972e24

Thought: I need to multiply this by 2.
Action: calculate: 5.972e24 * 2
PAUSE

You will be called again with this:

Observation: 1.1944e25

If you have the answer, output it as the Answer.

Answer: The mass of Earth times 2 is 1.1944e25.
"""

###################################################################################################################################
tools={"calculate": calculate, "currency_converter": currency_converter, "ddg_search": ddg_search, "ddg_news": ddg_news}
###################################################################################################################################
import os
os.system('clear')
# Initializing the LLM
llm = GroqLLM(api_key=os.environ.get("GROQ_API_KEY"))
# Creating an agent instance and pass the LLM instance to it
agent = Agent(llm=llm, cot_example=cot_example, tools=tools)
# Starting the agent's interaction loop
#agent.run(query='calculate the mass of two heaviest planets')
query = """Going for shopping, you go and shop 12 bananas and 5 apple. The price of bana per unit is .5 USD. Total bill is 10 USD. What is apple per unit in PKR?"""
print(query)
print(100*"-")
agent.run(query=query)