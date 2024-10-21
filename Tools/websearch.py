from googlesearch import search
import json

def free_search(query: str):
    '''Searches the internet for a given topic and returns relevant results.'''
    results = []
    
    # Perform a Google search using the googlesearch-python package
    for j in search(query, advanced=True, num_results=5, sleep_interval=10):
        result_dict = {
            "url": j.url,
            "title": j.title,
            "description": j.description
        }
        results.append(result_dict)

    # Return results as a formatted JSON string
    return json.dumps(results, indent=4)

if __name__ == "__main__":
    print(free_search("AI in Aviation"))
