
import requests
from bs4 import BeautifulSoup
#from dotenv import load_dotenv
#x = load_dotenv()
#print("Environment variables loaded successfully") if x else print("Error: Failed to load environment variables.")
###################################################################################################################
def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression and return the result as a float.    
    :param expression: A string representing the mathematical expression to evaluate.
    :return: The result of the evaluation as a float. If an error occurs during evaluation, returns NaN.
    """
    try:
        print('-> TOOL-CALCULATE CALLED')
        return float(eval(expression))
    except Exception as e:
        # Return NaN (Not a Number) in case of an error
        return float('nan')
###################################################################################################################
import requests
from typing import Any

def currency_converter(amount:Any, source_curr:str="USD", target_curr:str="GBP"):
    """
    Converts an amount from a source currency to a target currency.
    :param amount: The amount in the source currency.
    :param source_curr: The source currency code (e.g., 'USD').
    :param target_curr: The target currency code (e.g., 'EUR').
    :return: A formatted string with the converted amount.
    '''Example: currency_converter(100, 'USD', 'EUR')
    '100.00 USD is equivalent to: 80.00 EUR' '''
    """
    # A free API for currency conversion
    url = f"https://api.exchangerate-api.com/v4/latest/{source_curr}"
    
    response = requests.get(url)
    data = response.json()
    
    # Checking if the target currency is available
    if target_curr not in data["rates"]:
        return f"Error: Target currency '{target_curr}' not available in the exchange rates."
    amount = float(amount)
    # Calculating the conversion
    conv = data["rates"][target_curr] * amount

    print('-> TOOL-CURRENCY_CONVERTER CALLED')
    return conv
    #return f'{amount:.2f} {source_curr} is equivalent to: {conv:.2f} {target_curr}'
####################################################################################################################
import json
from duckduckgo_search import DDGS
from typing import Optional, Any
# Default headers for requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# DuckDuckGo search function
def ddg_search(query: str, max_results = "10", headers: Optional[Any] = None, timeout: Optional[int] = 10) -> str:
    """
    Search DuckDuckGo for a query and return the results.

    :param query: The query to search for.
    :param max_results: The maximum number of results to return (default=10).
    :param headers: Optional headers for the request. If not provided, defaults to DEFAULT_HEADERS.
    :param timeout: Optional timeout for the request (default=10 seconds).
    :return: A JSON string containing the search results.
    """
    headers = headers or DEFAULT_HEADERS
    ddgs = DDGS(headers=headers, timeout=timeout)
    results = ddgs.text(keywords=query, max_results=int(max_results)) 
    return json.dumps(results, indent=2)
####################################################################################################################
# DuckDuckGo news function
def get_news(topic: str, max_results = "10") -> str:
    """
    Search DuckDuckGo for the latest news based on a query and return the results.
    :param topic: The query to search for news.
    :param max_results: The maximum number of news results to return (default=10).
    :return: A JSON string containing the news results.
    """
    headers = DEFAULT_HEADERS
    ddgs = DDGS(headers=headers, timeout=60)
    results = ddgs.news(keywords=topic, max_results=int(max_results))
    return json.dumps(results, indent=2)
####################################################################################################################
# Example Usage:
if __name__ == "__main__":
    import os
    #os.system('clear')
    # Example: Search for web results
    search_results = ddg_search("what daye is today in USA?")
    print("Search_results\n",14*'-')
    print(search_results)

    # Example: Search for news
    news_results = get_news("AI breakthroughs")
    print("News_results\n",14*'-')
    print(news_results)
    
    conversion =  currency_converter(100, 'USD', 'EUR')
    print("Conversion:\n",conversion)
    
