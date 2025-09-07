import requests

def curr_conv(amount, source_curr, target_curr):
    """
    Convert an amount from a source currency to a target currency.

    :param amount: The amount in the source currency.
    :param source_curr: The source currency code (e.g., 'USD').
    :param target_curr: The target currency code (e.g., 'EUR').
    :return: A formatted string with the converted amount.
    """
    # Replace with your actual API URL and API key
    url = f"https://api.exchangerate-api.com/v4/latest/{source_curr}"
    
    response = requests.get(url)
    data = response.json()
    
    # Check if the target currency is available
    if target_curr not in data["rates"]:
        return f"Error: Target currency '{target_curr}' not available in the exchange rates."
    
    # Calculate the conversion
    conv = data["rates"][target_curr] * amount
    
    return f'{amount:.2f} {source_curr} is equivalent to: {conv:.2f} {target_curr}'

if __name__ == "__main__":
    amount = 1
    source_currency = 'USD'
    target_currency = 'PKR'
    print(curr_conv(amount, source_currency, target_currency))
