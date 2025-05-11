import requests
import json

def get_free_openrouter_models(keyword=None):
    # OpenRouter API endpoint for models
    url = "https://openrouter.ai/api/v1/models"
    
    try:
        # Make the API request
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the JSON response
        models_data = response.json()
        
        # Extract model names that are free
        free_models = [
            model["id"] for model in models_data.get("data", [])
            if model.get("pricing", {}).get("prompt", "0") == "0"
            and model.get("pricing", {}).get("completion", "0") == "0"
        ]
        
        free_models = [model for model in free_models if keyword.lower() in model.lower()] if keyword else free_models
        return free_models
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []

def main():
    # Get the list of free models
    free_models = get_free_openrouter_models()
    
    if free_models:
        print("Free models hosted on OpenRouter:")
        for model in free_models:
            print(f"- {model}")
    else:
        print("No free models found or error occurred.")

if __name__ == "__main__":
    main()