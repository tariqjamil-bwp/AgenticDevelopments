import json
from duckduckgo_search import DDGS
from typing import Optional, Any

# Default headers for requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# DuckDuckGo search function
def ddg_search(query: str, max_results: int = 5, headers: Optional[Any] = None, timeout: Optional[int] = 10) -> str:
    """
    Search DuckDuckGo for a query and return the results.

    :param query: The query to search for.
    :param max_results: The maximum number of results to return (default=5).
    :param headers: Optional headers for the request. If not provided, defaults to DEFAULT_HEADERS.
    :param timeout: Optional timeout for the request (default=10 seconds).
    :return: A JSON string containing the search results.
    """
    headers = headers or DEFAULT_HEADERS
    ddgs = DDGS(headers=headers, timeout=timeout)
    results = ddgs.text(keywords=query, max_results=max_results)
    return json.dumps(results, indent=2)


# DuckDuckGo news function
def ddg_news(query: str, max_results: int = 5, headers: Optional[Any] = None, timeout: Optional[int] = 10) -> str:
    """
    Search DuckDuckGo for the latest news based on a query and return the results.

    :param query: The query to search for news.
    :param max_results: The maximum number of news results to return (default=5).
    :param headers: Optional headers for the request. If not provided, defaults to DEFAULT_HEADERS.
    :param timeout: Optional timeout for the request (default=10 seconds).
    :return: A JSON string containing the news results.
    """
    headers = headers or DEFAULT_HEADERS
    ddgs = DDGS(headers=headers, timeout=timeout)
    results = ddgs.news(keywords=query, max_results=max_results)
    return json.dumps(results, indent=2)

# Example Usage:
if __name__ == "__main__":

    # Example: Search for web results
    search_results = ddg_search("what daye is today in Pakistan in Gregorian calender", max_results=3)
    print(search_results)

    # Example: Search for news
    news_results = ddg_news("AI breakthroughs", max_results=2)
    #print(news_results)
