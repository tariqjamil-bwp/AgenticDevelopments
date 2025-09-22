#Reference: https://openrouter.ai/docs/quickstart
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


def get_free_model(keyword:str='gemini'):
    MODELS = get_free_openrouter_models(keyword)
    print(MODELS)

    import os
    from google.adk.models.lite_llm import LiteLlm

    BASE_URL = "https://openrouter.ai/api/v1"
    API_KEY=os.getenv("OPENROUTER_API_KEY")
    MODEL = MODELS[1]
    model = LiteLlm(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    return model, MODEL
# Some other free models on 26th March:
# https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free
# https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free
def main():
    # Get the list of free models
    free_models = get_free_openrouter_models('gemini')
    
    if free_models:
        print("Free models hosted on OpenRouter:")
        for model in free_models:
            print(f"- {model}")
    else:
        print("No free models found or error occurred.")

if __name__ == "__main__":
    main()

