# adk_travel_planner.py

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
import asyncio
from dotenv import load_dotenv
from textwrap import dedent
import uuid

# --- Google ADK Imports ---
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

# =============================================================================
# 1. SETUP: CONFIGURATION AND CONSTANTS
# =============================================================================
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# --- Model Configuration ---
MODEL_NAME = "gemini/gemini-1.5-flash"

# --- ADK Configuration ---
APP_NAME = "travel_planner_app"
USER_ID = "traveler_001"

# =============================================================================
# 2. INITIALIZING THE LLM
# =============================================================================
AGENT_LLM = LiteLlm(model=MODEL_NAME)

# =============================================================================
# 3. DEFINING THE AGENTS
# =============================================================================

# --- Agent 1: Destination Expert ---
destination_expert = LlmAgent(
    name="Destination_Expert_Agent",
    model=AGENT_LLM,
    description="Analyzes user preferences to suggest a single best travel destination.",
    instruction=dedent("""
    You are the Destination Expert, a specialist in global travel destinations. Your responsibilities include:
    1. Analyzing user preferences (e.g., climate, activities, culture) to suggest suitable destinations.
    2. Providing detailed information about the recommended location, including attractions, best times to visit, and local customs.
    3. Considering factors like seasonality, events, and travel advisories in your recommendations.
    4. **SELECT 1 destination** that you think is the best choice for the user.
    5. **DO NOT CREATE AN ITINERARY.**
    Base your suggestions on a wide range of global destinations and current travel trends.
    Format your response with a clear "DESTINATION SUMMARY" header.
    """)
)

# --- Agent 2: Itinerary Creator ---
itinerary_creator = LlmAgent(
    name="Itinerary_Creator_Agent",
    model=AGENT_LLM,
    description="Crafts a detailed, day-by-day travel itinerary.",
    instruction=dedent("""
    You are the Itinerary Creator. You will receive user preferences and a destination summary. Your role is to:
    1. Create a day-by-day schedule for the entire trip duration mentioned by the user.
    2. Incorporate user preferences for activities, pace of travel, and must-see attractions from the context.
    3. Balance tourist attractions with local experiences.
    4. Consider practical aspects like travel times, opening hours, and meal times.
    Aim to create an engaging, well-paced itinerary.
    Format your response with a clear "ITINERARY" header, followed by a day-by-day breakdown.
    """)
)

# --- Agent 3: Budget Analyst ---
budget_analyst = LlmAgent(
    name="Budget_Analyst_Agent",
    model=AGENT_LLM,
    description="Analyzes the user's budget and provides a detailed cost breakdown.",
    instruction=dedent("""
    You are the Budget Analyst, an expert in travel budgeting. You will receive user preferences, a destination, and an itinerary. Your tasks are to:
    1. Analyze the user's overall budget for the trip.
    2. Provide detailed cost estimates for transportation, accommodation, food, and activities based on the itinerary.
    3. Suggest ways to optimize spending and find cost-effective options.
    4. Create a complete budget breakdown.
    Always strive for accuracy in your estimates and provide practical financial advice.
    Format your response with a clear "BUDGET" header.
    """)
)

# --- Agent 4: Report Writer (The Final Agent) ---
report_writer = LlmAgent(
    name="Report_Writer_Agent",
    model=AGENT_LLM,
    description="Compiles all information into a final, comprehensive travel report.",
    instruction=dedent("""
    You are the Report Compiler agent. You will receive a complete conversation containing a destination summary, a detailed itinerary, and a budget analysis.
    Your task is to compile all this information into a single, polished, and comprehensive travel report.

    **Report Structure:**
    - **Introduction:** A brief welcome and overview of the planned trip.
    - **Destination Summary:** Details from the Destination_Expert_Agent.
    - **Cultural Tips:** Information on local customs and etiquette.
    - **Itinerary:** The detailed day-by-day plan from the Itinerary_Creator_Agent.
    - **Transportation:** A summary of transport modes and their prices.
    - **Budget Breakdown:** The cost estimates and financial advice from the Budget_Analyst_Agent.
    - **Packing List:** A list of essential and optional items to pack.
    - **Conclusion:** A summary and final recommendations.

    Your final output must be this complete, well-structured report and nothing else.
    """)
)

# =============================================================================
# 4. DEFINING THE SEQUENTIAL WORKFLOW (The ADK "Crew")
# =============================================================================
travel_planner_crew = SequentialAgent(
    name='TravelPlannerCrew',
    description="A sequential crew of agents that plan a complete holiday trip.",
    sub_agents=[
        destination_expert,
        itinerary_creator,
        budget_analyst,
        report_writer,
    ],
)

# =============================================================================
# 5. MAIN EXECUTION FUNCTION
# =============================================================================
async def run_travel_plan(user_input: str):
    """
    Runs the full travel planning workflow for a given user query.
    """
    session_service = InMemorySessionService()
    session_id = f"travel_session_{uuid.uuid4()}" # Unique session for each run
    print("\nPlanning your trip... This may take a moment.")

    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    runner = Runner(agent=travel_planner_crew, app_name=APP_NAME, session_service=session_service)

    # The initial message contains the user's full request for the first agent.
    initial_message = types.Content(role='user', parts=[types.Part(text=user_input)])
    final_report = None

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=initial_message):
            print(f"...{event.author} is working on it...")
            if event.is_final_response():
                final_report = event.content.parts[0].text
                break
    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        return None

    return final_report

# =============================================================================
# 6. SCRIPT ENTRY POINT (Interactive Loop)
# =============================================================================
if __name__ == "__main__":
    # Ensures that file paths are relative to the script's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Welcome to the ADK Travel Planning Assistant!")
    print("Plan your trip by providing details like destination ideas, duration, budget, and interests.")
    print("Example: 'Plan a 7-day trip to Italy for a budget of $2000. I love history and food.'")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_input = input("\nEnter your trip details: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Have a great day!")
            break

        try:
            final_result = asyncio.run(run_travel_plan(user_input))
            if final_result:
                print("\n------------- Final Holiday Details ----------\n")
                print(final_result)
                print("\n--------------------------------------------\n")
            else:
                print("\n⚠️ The travel plan could not be generated. Please try again.")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("Please try again.")