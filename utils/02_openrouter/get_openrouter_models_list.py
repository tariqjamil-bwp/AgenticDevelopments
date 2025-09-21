import requests
import logging
import re
from typing import List, Dict, Any, Optional

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for the API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def get_free_models_with_capabilities(
    keyword: Optional[str] = None, 
    tool_use: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Fetches details for free models from OpenRouter, with powerful filtering.
    It intelligently guesses the parameter count from the model ID using the
    specific pattern "-<number>b" if not provided by the API.

    Args:
        keyword (Optional[str]): Filters models by a keyword in the ID or name.
        tool_use (Optional[bool]): Filters models by their tool-use capability.

    Returns:
        A filtered list of dictionaries for each model.
    """
    url = f"{OPENROUTER_BASE_URL}/models"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        all_models_data = response.json().get("data", [])
        
        filtered_models = []
        for model in all_models_data:
            # Stage 1: Skip non-free models
            if model.get("pricing", {}).get("prompt") != "0" or model.get("pricing", {}).get("completion") != "0":
                continue

            # Stage 2: Extract capabilities and parameters
            architecture = model.get("architecture", {})
            description = model.get("description", "").lower()
            model_id = model.get("id", "")

            # Determine tool support
            explicit_support = architecture.get("tool_use", False) or architecture.get("function_calling", False)
            inferred_support = "tool use" in description or "function calling" in description
            supports_tools = explicit_support or inferred_support

            # --- REFINED: Smart Parameter Count Logic ---
            params_in_b = 0.0
            param_source = "N/A"
            
            # 1. Try to get official data first
            n_params = architecture.get("n_parameters")
            if n_params and n_params > 0:
                params_in_b = round(n_params / 1_000_000_000, 1)
                param_source = "API"
            else:
                # 2. Fallback to guessing with the refined, more precise regex
                # This pattern specifically looks for a hyphen, a number, and a 'b'.
                # e.g., "-7b", "-70b", "-1.5b"
                match = re.search(r"-(\d+\.?\d*)b", model_id, re.IGNORECASE)
                if match:
                    try:
                        # group(1) captures the number between the hyphen and 'b'.
                        params_in_b = float(match.group(1))
                        param_source = "Guessed"
                    except ValueError:
                        # This is unlikely to fail with this regex, but it's safe to have.
                        pass 

            model_details = {
                "id": model_id,
                "name": model.get("name", ""),
                "context_length": model.get("context_length", 0),
                "supports_tools": supports_tools,
                "parameters_in_b": params_in_b,
                "param_source": param_source,
            }

            # Stage 3: Apply user's filters
            keyword_match = (keyword is None) or (keyword.lower() in model_details['id'].lower()) or (keyword.lower() in model_details['name'].lower())
            tool_use_match = (tool_use is None) or (model_details['supports_tools'] is tool_use)

            if keyword_match and tool_use_match:
                filtered_models.append(model_details)
                
        return filtered_models
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching models from OpenRouter: {e}")
        return []

# --- Helper function to print results (no changes needed here) ---
def print_model_list(title: str, models: List[Dict[str, Any]]):
    print("\n" + "="*85)
    print(title)
    print("="*85)
    if not models:
        print("No models found matching the criteria.")
        return
        
    models.sort(key=lambda m: (m.get('parameters_in_b', 0), m.get('context_length', 0)), reverse=True)
    
    for model in models:
        tool_icon = "✅" if model['supports_tools'] else "❌"
        params_b = model.get('parameters_in_b', 0)
        source = model.get('param_source', 'N/A')
        params_str = f"{params_b}B ({source})" if params_b > 0 else "N/A"
        
        print(
            f"- {model['id']:<40} | "
            f"Tools: {tool_icon} | "
            f"Params: {params_str:<15} | "
            f"Context: {model['context_length']:,}"
        )
    print(f"\nFound {len(models)} models.")

import requests
import json

def get_free_openrouter_models(keywords=None):
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
        
        # Handle keyword as string or list
        if isinstance(keywords, str):
            keywords = keywords.split()  # Split by whitespace, or use keyword.split(',') for comma-separated
        if keywords:
            free_models = [model for model in free_models if any(kw.lower() in model.lower() for kw in keywords)]
        
        return free_models
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []

def main():
    
    # Example 1: Get the list of free models
    free_models = get_free_openrouter_models(keywords='gemini deepseek')  # or keyword="gemini llama" or keyword="gemini,llama"
    
    if free_models:
        print("Free models hosted on OpenRouter:")
        for model in free_models:
            print(f"- {model}")
    else:
        print("No free models found or error occurred.")

    # Example 2: Show all free models with the refined guessing logic
    all_free_models = get_free_models_with_capabilities()
    print_model_list("Query: All free models (Sorted by Params, then Context)", all_free_models)

    # Example 3: Find all tool-using models with 'gemma' in the name
    gemma_tool_models = get_free_models_with_capabilities(keyword="gemma", tool_use=True)
    print_model_list("Query: 'gemma' models WITH Tool Support", gemma_tool_models)


if __name__ == "__main__":
    main()
    