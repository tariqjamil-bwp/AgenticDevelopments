from typing import Union, Any, Optional, List, Dict, Annotated
import requests
import os
import json
from sympy import sympify
from duckduckgo_search import DDGS
from jinja2 import Template
from colorama import Fore, Style
import re

########################################################################################################################

def calculate(
    expression: Annotated[str, "Mathematical expression to evaluate"]
) -> Union[float, str]:
    """
    Evaluates a mathematical expression using sympy and returns the result as a float.
    """
    print(' -> calculate Tool Called --\n')
    try:
        result = sympify(expression)
        print(result)
        return float(result)
    except Exception as e:
        print(f"Error: {e}")
        return "NaN"

########################################################################################################################

def currency_converter(
    amount: Annotated[float, "Amount in source currency"],
    source_curr: Annotated[str, "Source currency code"] = "USD",
    target_curr: Annotated[str, "Target currency code"] = "GBP"
) -> str:
    """
    Converts an amount from a source currency to a target currency using an API.
    """
    print(' -> currency_converter Tool Called --\n')
    url = f"https://api.exchangerate-api.com/v4/latest/{source_curr}"
    
    response = requests.get(url)
    data = response.json()
    
    if target_curr not in data["rates"]:
        return f"Error: Target currency '{target_curr}' not available in the exchange rates."
    
    conv = data["rates"][target_curr] * amount
    print('-> TOOL-CURRENCY_CONVERTER CALLED')
    return f'{amount:.2f} {source_curr} is equivalent to: {conv:.2f} {target_curr}'

########################################################################################################################

def ddg_search(
    query: Annotated[str, "Search query for DuckDuckGo"],
    max_results: Annotated[Optional[int], "Maximum number of results to retrieve"] = 4,
    timeout: Annotated[Optional[int], "Timeout for the request in seconds"] = 10
) -> str:
    """
    Searches the web for a query using DuckDuckGo and returns the results.
    """
    print(' -> ddg_search Tool Called --\n')
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    ddgs = DDGS(headers=headers, timeout=timeout)
    results = ddgs.text(keywords=query, max_results=int(max_results)) 
    return json.dumps(results, indent=2)

########################################################################################################################

def get_news(
    topic: Annotated[str, "Topic for news search"],
    max_results: Annotated[int, "Maximum number of news results to return"] = 4
) -> str:
    """
    Retrieves the latest news based on a specified topic using DuckDuckGo.
    """
    print(' -> get_news Tool Called --\n')
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    ddgs = DDGS(headers=headers, timeout=60)
    results = ddgs.news(keywords=topic, max_results=int(max_results))
    return json.dumps(results, indent=2)

########################################################################################################################

def get_weather(
    location: Annotated[str, "Location name for weather information"]
) -> str:
    """
    Retrieves the current weather (temoerature, humidity, rain, date, time etc) for a specified location.
    """
    print(' -> get_weather Tool Called --\n')
    
    if not location:
        return "Location cannot be empty. Please provide a valid location."
    
    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        return "Missing API key. Please set WEATHER_API_KEY environment variable."
    
    API_params = {
        "key": api_key,
        "q": location.strip(),
        "aqi": "yes",
        "alerts": "no",
    }
    print('*')
    try:
        response = requests.get("http://api.weatherapi.com/v1/current.json", params=API_params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    
    if response.status_code != 200:
        return f"Error: Unable to fetch weather data for {location}. Status code: {response.status_code}"
    
    try:
        str_response: str = json.dumps(response.json())
        print('***')
    except ValueError:
        return "Error: Unable to parse weather data."
    return str_response

########################################################################################################################

def tavily_search(
    query: Annotated[str, "Search query for Tavily API"],
    max_results: Annotated[Optional[int], "Maximum number of results to retrieve"] = 5,
    timeout: Annotated[Optional[int], "Timeout for the API request in seconds"] = 10
) -> str:
    """
    Searches Tavily's API with a specified query and returns the results.
    """
    print(' -> tavily_search Tool Called --\n')
    
    try:
        api_url = "https://api.tavily.com/search"
        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            return "Error: Missing Tavily API key. Set TAVILY_API_KEY environment variable."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        params = {
            "query": query,
            "limit": max_results,
        }

        response = requests.get(api_url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        
        results = response.json()
        return json.dumps(results, indent=2)
    
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

########################################################################################################################

def get_tool_specifications(
    tools: Annotated[Dict[str, callable], "Dictionary of tool names and functions"],
    llm: Annotated[callable, "LLM function to generate tool specifications"]
) -> List[dict]:
    """
    Generates tool specifications for the provided tools and returns them as JSON objects.
    """
    body = """
    You are a helpful assistant familiar with OpenAI tool specifications, which have the following JSON format:

    {
        "type": "function",
        "function": {
            "name": "<function_name>",
            "description": "<function_description>",
            "parameters": {
                "type": "object",
                "properties": {
                    "<parameter_name>": {
                        "type": "<parameter_type>",
                        "description": "<parameter_description>"
                    }
                },
                "required": ["<parameter_name>"]
            }
        }
    }

    Generate an OpenAI Tools Specification compatible JSON string for the following tool:
    Tool Name: {{ tool_name }}
    Tool Description: {{ tool_doc }}
    
    Please respond only with the JSON string, without any additional text.
    """

    spec = []

    for tool_name, tool_func in tools.items():
        prompt = Template(body).render(tool_name=tool_name, tool_doc=tool_func.__doc__.strip())
        response = llm(message=prompt)
        
        try:
            json_response = json.loads(response.strip())
            spec.append(json_response)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON for tool '{tool_name}': {e}")
            continue

    return spec

########################################################################################################################

def cprint(
    message: Annotated[str, "Message to print with color formatting"]
):
    """
    Prints colored output based on the label followed by a colon.
    """
    color_mapping = {
        "Agent": Fore.BLUE+Style.BRIGHT,
        "Thought": Fore.CYAN,
        "Action": Fore.YELLOW,
        "Pause": Fore.MAGENTA,
        "Observation": Fore.GREEN,
        "Answer": Fore.BLUE,
    }
    
    lines = message.splitlines()
    last_color = Style.RESET_ALL

    for line in lines:
        label_match = re.match(r"(\w+):", line)
        if label_match:
            label = label_match.group(1)
            color = color_mapping.get(label, Style.RESET_ALL)
            last_color = color
            print(f"{color}{line}{Style.RESET_ALL}")
        else:
            print(f"{last_color}{line}{Style.RESET_ALL}")

########################################################################################################################

# Example Usage:
if __name__ == "__main__":
    import os
    os.system('clear')
    # Example: Search for web results
    print(get_weather("Abuja, Nigeria"))

    search_results = ddg_search("what day is today in USA?")
    print("Search_results\n", 14 * '-')
    print(search_results)

    # Example: Search for news
    news_results = get_news("AI breakthroughs")
    print("News_results\n", 14 * '-')
    print(news_results)
    
    conversion = currency_converter(100.0, 'USD', 'EUR')
    print("Conversion:\n", conversion)

    print(calculate("sqrt(3)*exp(4)+5"))
