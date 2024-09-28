from langchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from setup_environment import set_environment_variables

#openai_llm = ChatOpenAI(temperature=0.4)
# Import required dependencies
from crewai import Crew, Agent, Task
from textwrap import dedent
import os
import json
import requests
##########################################################################################################
from utils import GroqChatLLM

llm = GroqChatLLM()
set_environment_variables("Email_&_Weather")
##########################################################################################################
class Agents:
	def classifierAgent():
	    return Agent(
	      role='Email Classifier',
	      goal='You will be given an email and you have to classify the given email in one of these 2 categories: 1) Important 2) Casual ',
	      backstory='An email classifier who is expert in classifying every type of email and have classified so many emails so far',
	      verbose = True,
	      allow_delegation=False,
          llm=llm
	    )
	def emailWriterAgent():
	  return Agent(
	    role='Email writing expert',
	    goal="You are email writing assistant for Shivam. You will be given an email and a category of that email and your job is to write a reply for that email. If email category is 'Important' then write the reply in professional way and If email category is 'Casual' then write in a casual way",
	    backstory='An email writer with an expertise in email writing for more than 10 years',
	    verbose = True,
	    allow_delegation=False,
        llm=llm
	  )
class Tasks:
	def classificationTask(agent,email):
	    return Task(
	        description=dedent(f"""
	        You have given an email and you have to classify this email
	        {email}
	        """),
	        agent = agent,
	        expected_output = "Email category as a string"
	    )
	def writerTask(agent,email):
	  return Task(
	      description=dedent(f"""
	      Create an email response to the given email based on the category provided by 'Email Classifier' Agent
	      {email}
	      """),
	      agent = agent,
	      expected_output = "A very concise response to the email based on the category provided by 'Email Classifier' Agent"
	  )
class EmailCrew:
  def __init__(self,email):
    self.email = email
  def run(self):
		# Agents
    classifierAgent = Agents.classifierAgent()
    writerAgent = Agents.emailWriterAgent()
		# Tasks
    classifierTask = Tasks.classificationTask(agent=classifierAgent,email=self.email)
    writerTask = Tasks.writerTask(agent=writerAgent,email=self.email)
		# Create crew
    crew = Crew(
      agents=[classifierAgent,writerAgent],
      tasks=[classifierTask,writerTask],
      verbose=2, # You can set it to 1 or 2 to different logging levels
    )
		# Run the crew
    result = crew.kickoff()
    return result
##########################################################################################################
from langchain.tools import tool
#weather = OpenWeatherMapAPIWrapper()
from tools import get_weather
class Tools:
  @tool("Tool to get the weather of any location")
  def weather_tool(location):
    """
    Use this tool when you have given a location and you want to find the weather of that location
    """
    #data = weather.run(location)
    data = get_weather(location)
    return data
##########################################################################################################  
class Agents:
	# ... Other agents
    def weatherAgent():
        return Agent(
            role = 'Weather Expert',
            goal = 'You will be given a location name and you have to find the weather information about that location using the tools provided to you',
            backstory = "An weather expert who is expert in providing weather information about any location",
            tools = [Tools.weather_tool],
            verbose = True,
            allow_delegation = False,
            llm=llm
            )
class Tasks:
	# ... Other tasks
    def weatherTask(agent,query):
        return Task(
            description = dedent(f"""
            Get the location from the user query and find the weather information about that location

            Here is the user query:
            {query}
            """),
            agent = agent,
            expected_output = "A weather information asked by user"
        )

##########################################################################################################
from typing import TypedDict
class AgentState(TypedDict):
    messages: list[str]
    email: str
    query: str
    category: str
##########################################################################################################
class Nodes:
  def writerNode(self,state):
    email = state["email"]
    emailCrew = EmailCrew(email)
    crewResult = emailCrew.run()
    messages = state["messages"]
    messages.append(crewResult)
    return {"messages": messages}

  def weatherNode(self,state):
    query = state["query"]
    weatherAgent = Agents.weatherAgent()
    weatherTask = Tasks.weatherTask(agent=weatherAgent,query=query)
    result = weatherTask.execute()
    messages = state["messages"]
    messages.append(result)
    return {"messages": messages}

  def replyNode(self,state):
	  query = state["query"]
	  agent = openai_llm.invoke(f"""
	    {query}
	  """)
	  messages = state["messages"]
	  messages.append(agent.content)
	  return {"messages": messages}
  def entryNode(self,state):
    input = state["query"]
    agent = llm.invoke(f"""
      User input
      ---
      {input}
      ---
      You have given one user input and you have to perform actions on it based on given instructions

      Categorize the user input in below categories
      email_query: If user want to generate a reply to given email
      weather_query: If user want any weather info about given location
      other: If it is any other query

      After categorizing your final RESPONSE must be in json format with these properties:
      category: category of user input
      email: If category is 'email_query' then extract the email body from user input with proper line breaks and add it here else keep it blank
      query: If category is 'weather_query' or 'other' then add the user's query here else keep it blank
    """)
    response = json.loads(agent.content)
    return {'email': response["email"], 'query': response['query'], 'category': response['category']}
#########################################################################################################
def where_to_go(state):
    cat = state['category']
    print("Category: ",cat)
    if cat == "email_query":
        return "email"
    elif cat == "weather_query":
        return "weather"
    else:
        return "reply"
#########################################################################################################
from langgraph.graph import Graph, END, StateGraph
workflow = StateGraph(AgentState)
node = Nodes()
workflow.add_node('entryNode',  node.entryNode)
workflow.add_node('weatherNode',node.weatherNode)
workflow.add_node("responder",  node.replyNode)
workflow.add_node('emailNode',  node.writerNode)

workflow.add_conditional_edges('entryNode',where_to_go,{
    "email":   "emailNode",
    "weather": "weatherNode",
    "reply":    "responder"
    })

workflow.add_edge("weatherNode",END)
workflow.add_edge("responder",  END)
workflow.add_edge("emailNode",  END)

workflow.set_entry_point("entryNode")
app = workflow.compile()

##########################################################################################################
query = """
Can you reply to this email

Hello,
Thank you for applying to xyz company
can you share me your previous CTC
Thanks,
HR
"""
inputs = {"query": query, "messages": [query]}
result = app.invoke(inputs)
print("Agent Response:",result['messages'][-1])