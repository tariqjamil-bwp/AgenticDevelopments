#from crewai.tools import tool
import requests
#@tool('free_search')
def free_search(query: str):
    '''Searches the internet for a given topic and returns relevant results. There is only 1 parameter which is just the query string'''
    url = f"{query} - Google Search"

    response = requests.get(url)
    results = []
    for j in search(query, advanced=True, num_results=5, sleep_interval=10):
        result_dict = {
            "url": j.url,
            "title": j.title,
            "description": j.description
        }
        results.append(result_dict)

    return json.dumps(results, indent=4)

if __name__ == "__main__":
    print(free_search("hello"))