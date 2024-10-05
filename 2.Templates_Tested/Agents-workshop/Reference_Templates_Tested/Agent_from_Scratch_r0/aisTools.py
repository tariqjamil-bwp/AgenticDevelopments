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
def get_planet_mass(planet_name: str) -> float:
    """
    Retrieve the mass of a specified planet in kilograms.
    :param planet_name: The name of the planet for which to retrieve the mass. The name is case-insensitive.
    :return: The mass of the specified planet in kilograms as a float. If the planet is not found, returns NaN.
    """
    planet_masses = {
        "mercury": 3.3011e23,
        "venus":   4.8675e24,
        "earth":   5.9723e24,
        "mars":    6.4171e23,
        "jupiter": 1.8982e27,
        "saturn":  5.6834e26,
        "uranus":  8.6810e25,
        "neptune": 1.02413e26
    }
    print('-> TOOL-GET_PLANET_MASS CALLED')
    return planet_masses.get(planet_name.lower(), float('nan'))
###################################################################################################################
import requests
def currency_converter(amount, source_curr, target_curr):
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
    return f'{amount:.2f} {source_curr} is equivalent to: {conv:.2f} {target_curr}'
####################################################################################################################
import requests

def ddg_search(query: str) -> str:
    """
    Perform a search using DuckDuckGo and return the top result.
    
    :param query: The search query to perform.
    :return: A string containing the title and URL of the top result, or an error message if none is found.
    """
    url = "https://api.duckduckgo.com"
    params = {
        "q": query,
        "format": "json",
        "pretty": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    print('-> TOOL-DDG_SEARCH CALLED')
    print(response)
    if "AbstractText" in data and data["AbstractText"]:
        return f"Result: {data['AbstractText']}"
    elif "RelatedTopics" in data and len(data["RelatedTopics"]) > 0:
        return f"Top result: {data['RelatedTopics'][0]['Text']} ({data['RelatedTopics'][0]['FirstURL']})"
    else:
        return "No relevant results found."

    ####################################################################################################################
if __name__ == "__main__":
        
    # Example usage
    amount = 3.80
    source_currency = 'USD'
    target_currency = 'PKR'
    #print(currency_converter(amount, source_currency, target_currency))
    x = ddg_search("what is today?")
    print(x)