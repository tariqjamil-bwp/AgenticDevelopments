import os
from typing import List
# Import the necessary libraries
from google import genai
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
load_dotenv(override=True)
# #############################################################################
# --- Main Checker Function ---
# #############################################################################
# This is the list of models we want to check.
models_to_check: List[str] = [
        # Recommended Stable Models (Most likely in Free Tier)
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash',
        
        # Newer Generation Models
        #'gemini-2.5-pro',
        'gemini-2.5-flash',
        
        # Older Generation Models
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
        
        # Preview / Specialized Models (Less likely to be standard free tier)
        'gemini-2.5-flash-lite-preview-06-17',
        'gemini-2.0-flash-preview-image-generation',
        ]    

def get_available_gemini_models():
    """
    Checks a predefined list of Google Gemini models against your API key to
    determine their availability and inferred free-tier status.
    """
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # --- 1. Setup and Configuration ---
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    console = Console()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    if not api_key:
        console.print("[bold red]ERROR:[/bold red] GOOGLE_API_KEY not found in .env file. Halting.")
        return 
    # Simple prompt for testing
    prompt = "What is the capital of France?"
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # --- 2. Create Results Table and Query Models ---
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    table = Table(title="Gemini Model Accessibility Test")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Details / Response", style="magenta")

    console.print(f"Sending test prompt: '[italic]{prompt}[/italic]'\n")
    available_models = []
    for model_name in models_to_check:
        status = ""
        details = ""
        try:
            
            response = client.models.generate_content(
                model=model_name, 
                contents=prompt
                )
            # If we get here, it was successful
            status = "[bold green]✅ SUCCESS[/bold green]"
            details = response.text.strip()
            table.add_row(model_name, status, details)
            available_models.append(model_name)
        except Exception as e:
            # If any error occurs, it's a failure
            #status = "[bold red]❌ FAILURE[/bold red]"
            # Provide a clean, readable error message
            error_message = str(e).split('\n')[0] # Get the first line of the error
            details = f"Error: {error_message}"    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # --- 3. Display Results ---
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    console.print(table)
    console.print("\n[italic]Note: Failures can be due to an invalid model name, lack of access with your API key, or temporary API issues.[/italic]")

    return available_models

# #############################################################################
# --- Script Entry Point ---
# #############################################################################
if __name__ == "__main__":
    get_available_gemini_models()
