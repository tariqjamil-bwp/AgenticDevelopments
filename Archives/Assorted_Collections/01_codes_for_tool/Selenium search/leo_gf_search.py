import os
os.chdir('1.Assorted_Collections/01_codes_for_tool/langchain_tools')

# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
#from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from tools.selenium_search import SeleniumSearchTool
from langchain.llms import Ollama

#llm = ChatOpenAI(temperature=0)
llm = Ollama(model='gemma2:2b')
search = SeleniumSearchTool()

# Load the tool configs that are needed.
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
# Load in the Selenium Search Tool first
tools = [search]

from pydantic import BaseModel, Field
class CalculatorInput(BaseModel):
    question: str = Field()

calc_tool = Tool.from_function(
    func=llm_math_chain.run,
    name="Calculator",
    description="useful for when you need to answer questions about math",
    args_schema=CalculatorInput,
    # coroutine= ... <- you can specify an async method if desired as well
    )

#tools = [search, calc_tool]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, 
    llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
agent.run("List verified websites from reuter, AFP, Aljazeera?")
