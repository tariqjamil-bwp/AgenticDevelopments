# Initialize global search tool
class WebSearchTool:
    """DuckDuckGo search tool for documentation and reference lookup."""
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search DuckDuckGo for relevant results."""
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Extract relevant results
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:max_results]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('Text', '').split(' - ')[0],
                            'snippet': topic.get('Text', ''),
                            'url': topic.get('FirstURL', ''),
                            'source': 'DuckDuckGo'
                        })
            
            # Add abstract if available
            if data.get('Abstract'):
                results.insert(0, {
                    'title': data.get('AbstractText', 'Overview'),
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', ''),
                    'source': data.get('AbstractSource', 'Wikipedia')
                })
            
            return results[:max_results]
            
        except Exception as e:
            print(f"Search error: {e}")
            return [{
                'title': 'Search Error',
                'snippet': f'Unable to search for "{query}": {str(e)}',
                'url': '',
                'source': 'Error'
            }]

# Create global search tool instance
search_tool = WebSearchTool()

def search_documentation(query: str, topic_type: str = "general") -> str:
    """
    Search for documentation and code examples related to the given query.
    
    Args:
        query: What to search for (e.g., "Python Gemini API tutorial")
        topic_type: Type of search - "api", "tutorial", "documentation", "example"
    
    Returns:
        Formatted search results with relevant information
    """
    # Enhance query based on topic type
    enhanced_queries = {
        "api": f"{query} API documentation python",
        "tutorial": f"{query} tutorial python example",
        "documentation": f"{query} official documentation",
        "example": f"{query} python code example github",
        "general": query
    }
    
    search_query = enhanced_queries.get(topic_type, query)
    results = search_tool.search(search_query, max_results=3)
    
    if not results:
        return f"No search results found for: {query}"
    
    formatted_results = f"Search Results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        formatted_results += f"**Result {i}:**\n"
        formatted_results += f"Title: {result['title']}\n"
        formatted_results += f"Source: {result['source']}\n"
        formatted_results += f"Summary: {result['snippet'][:200]}...\n"
        if result['url']:
            formatted_results += f"URL: {result['url']}\n"
        formatted_results += "\n" + "-"*50 + "\n\n"
    
    return formatted_results

