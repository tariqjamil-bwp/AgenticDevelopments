# Define the tool schema
tool_schema_get_news = {
    "type": "function",
    "function": {
        "name": "get_news",
        "description": "Search the web for the latest news based on a query and return the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The query to search for news."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of news results to return."
                }
            },
            "required": ["topic"]
        }
    }
}

tool_schema_ddg_search = {
    "type": "function",
    "function": {
        "name": "ddg_search",
        "description": "Search the web for the latest websites / URLs based on a query and return the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The query to search for web."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of URLs results to return."
                }
            },
            "required": ["topic"]
        }
    }
}

tool_schema_get_weather = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a specified location. This includes temperature, humidity, AQI, rain, snow, current time and data etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location name for weather information."
                }
            },
            "required": ["location"]
        }
    }
}

tool_schema_calculate = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluates a mathematical expression using sympy and returns the result as a float.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A string representing a mathematical expression."
                }
            },
            "required": ["expression"]
        }
    }
}

# Combine into a list
tools_spec = [
    tool_schema_get_news,
    tool_schema_ddg_search,
    tool_schema_get_weather,
    tool_schema_calculate
]

# Combine into a list
tools_spec = [
    tool_schema_get_news,
    tool_schema_ddg_search,
    tool_schema_get_weather,
    tool_schema_calculate
]

#generationn of schema
##########################################################################################################################
from Tools_r3 import calculate, currency_converter, get_news, ddg_search, get_weather
from Tools_r3 import get_tool_specifications
import os
import openai
from groq import Groq
from pprint import pprint
##########################################################################################################################

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

def call_llm(message, client=client):

    if not isinstance(message, list): # in case formatted message is not given
        messages=[
            # Set an optional system message. This sets the behavior of the
            {"role": "system", "content": "you are a helpful assistant."},
            # Set a user message for the assistant to respond to.
            {"role": "user", "content": message}
            ]
    else:
        messages=message
    
    chat_completion = client.chat.completions.create(
        messages=messages,            
        model="llama-3.2-90b-text-preview",
        #model='llama3-70b-8192',
        temperature=0,
        max_tokens=128,
        stream=False,
    )

    return chat_completion.choices[0].message.content
##########################################################################################################################
os.system("clear")

tools = {'get_news': get_news, 'get_weather': get_weather, "calculate": calculate}
spec = get_tool_specifications(tools, call_llm)
pprint(spec, width=160)


#tools_spec = [{'type': 'function', 'name': 'get_news', 'description': 'Search the web for the latest news based on a query and return the results.', 'parameters': {'type': 'object', 'properties': {'topic': {'type': 'string', 'description': 'The query to search for news.'}, 'max_results': {'type': 'integer', 'description': 'The maximum number of news results to return.'}}, 'required': ['topic']}}, {'type': 'function', 'name': 'get_weather', 'description': 'Get the current weather for a specified location. This includes temperature, humidity, AQI, rain, snow, current time and data etc.', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The location name for weather information.'}}, 'required': ['location']}}, {'type': 'function', 'function': {'name': 'calculate', 'description': 'Evaluates a mathematical expression using sympy and returns the result as a float.', 'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string', 'description': 'A string representing a mathematical expression.'}}, 'required': ['expression']}}}]
