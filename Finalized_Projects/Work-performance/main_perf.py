# task_manager_agent.py

# =============================================================================
# 0. IMPORTS AND SETUP
# =============================================================================
import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# --- Pydantic for Type Validation ---
from pydantic import BaseModel, Field

# --- Google ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm  # Import LiteLlm
from google.genai import types

# =============================================================================
# 1. CONFIGURATION AND ENVIRONMENT
# =============================================================================
# Loading environment variables from .env file
load_dotenv()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Loading configuration variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is not set. Please add it to your .env file.")

# Set the API key for LiteLLM to use
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# --- Default Settings ---
DEFAULT_MODEL = "gemini-1.5-flash"  # Model name, without the "gemini/" prefix
DEFAULT_APP_NAME = "task_management_app"
DEFAULT_USER_ID = "default_user"
DEFAULT_SESSION_ID = "default_session"

# =============================================================================
# 2. DEFINING TOOLS WITH INPUT VALIDATION
# =============================================================================

# --- Pydantic Models for Tool Inputs ---
class DurationInput(BaseModel):
    start_time: str = Field(description="Start time in HH:MM format", pattern=r"^\d{2}:\d{2}$")
    end_time: str = Field(description="End time in HH:MM format", pattern=r"^\d{2}:\d{2}$")

class CompletionInput(BaseModel):
    tasks: int = Field(description="Number of tasks", ge=0)
    time_per_task: int = Field(description="Time per task in minutes", ge=0)

class ProductivityInput(BaseModel):
    tasks_completed: int = Field(description="Number of tasks completed", ge=0)
    total_time: int = Field(description="Total time in minutes", gt=0)


# --- Tool Functions ---
def calculate_task_duration(start_time: str, end_time: str) -> str:
    """Calculates the duration between two times in HH:MM format."""
    try:
        # Validate inputs using Pydantic
        DurationInput(start_time=start_time, end_time=end_time)
        start = datetime.strptime(start_time, "%H:%M")
        end = datetime.strptime(end_time, "%H:%M")
        duration = end - start
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        return f"{hours} hours and {minutes} minutes"
    except Exception as e:
        return f"Invalid input error: {e}"

def estimate_task_completion(tasks: int, time_per_task: int) -> str:
    """Estimates the total time needed to complete a number of tasks."""
    try:
        CompletionInput(tasks=tasks, time_per_task=time_per_task)
        total_minutes = tasks * time_per_task
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours} hours and {minutes} minutes"
    except Exception as e:
        return f"Invalid input error: {e}"

def calculate_productivity(tasks_completed: int, total_time: int) -> str:
    """Calculates the tasks completed per hour."""
    try:
        ProductivityInput(tasks_completed=tasks_completed, total_time=total_time)
        hours = total_time / 60
        tasks_per_hour = tasks_completed / hours
        return f"{tasks_per_hour:.2f} tasks per hour"
    except Exception as e:
        return f"Invalid input error: {e}"

# =============================================================================
# 3. SETTING UP THE AGENT
# =============================================================================
def get_agent() -> LlmAgent:
    """Configures and returns an LlmAgent with task management tools."""
    # Use LiteLlm to connect to the Gemini model
    model_instance = LiteLlm(model=f"gemini/{DEFAULT_MODEL}")
    
    return LlmAgent(
        name="task_management_agent",
        model=model_instance,
        tools=[calculate_task_duration, estimate_task_completion, calculate_productivity],
        instruction=(
            "You are a task management assistant. Your goal is to analyze the user's query, "
            "extract the necessary parameters, call the appropriate tool, and return the final "
            "result clearly. If multiple calculations are needed, perform them step-by-step."
        ),
        description="Provides task duration, completion time, and productivity calculations."
    )

# =============================================================================
# 4. EXECUTING THE AGENT WORKFLOW
# =============================================================================
async def run_agent(agent: LlmAgent, query: str) -> list:
    """Executes a query using the agent and runner, returning the final response."""
    session_service = InMemorySessionService()
    session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    await session_service.create_session(
        app_name=DEFAULT_APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id
    )
    
    runner = Runner(
        agent=agent,
        app_name=DEFAULT_APP_NAME,
        session_service=session_service
    )
    
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_reply = "No response received."

    async for event in runner.run_async(user_id=DEFAULT_USER_ID, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content:
            final_reply = event.content.parts[0].text
            break
    
    return final_reply

# =============================================================================
# 5. SCRIPT ENTRY POINT
# =============================================================================
async def main():
    """Sets up the agent and executes an example query."""
    try:
        agent = get_agent()
        
        query = """
        My work stats for the week are:
        - Day 1: Worked from 09:00 to 17:00 and completed 8 tasks.
        - Day 2: Worked from 10:00 to 18:00 and completed 6 tasks.
        - Day 3: Worked from 11:00 to 19:00 and completed 4 tasks.
        - Day 4: Worked from 12:00 to 20:00 and completed 2 tasks.
        - Day 5: Worked from 13:00 to 20:00 and completed 10 tasks.
        
        Based on this, what was my average productivity rate for the week?
        Please show your work step-by-step and double-check your calculations.
        """
        
        print("\nTask Management Assistant")
        print("------------------------")
        print(f"Query: {query.strip()}")
        print("\nThinking...")
        
        response = await run_agent(agent, query)
        
        print("\nFinal Report:")
        print("-------------")
        print(response)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())