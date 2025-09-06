# -*- coding: utf-8 -*-
"""
Google ADK Conversion of a CrewAI Customer Support Multi-Agent System.

This script demonstrates how to convert a CrewAI workflow into the Google
Agent Development Kit (ADK) framework.

Original CrewAI concepts are mapped to ADK components:
- CrewAI Agent -> ADK LlmAgent
- CrewAI Task -> Integrated into the LlmAgent's instructions
- CrewAI Tool -> Wrapped in an async function and passed to an ADK FunctionTool
- CrewAI Crew -> ADK SequentialAgent
- CrewAI Memory -> Handled by the ADK SessionService
"""

import os
import asyncio
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

# Import the specific CrewAI Tool used in the notebook
from crewai_tools import ScrapeWebsiteTool

# -----------------------------
# Constants and API Key Checks
# -----------------------------
APP_NAME = "customer_support_app"
USER_ID = "support_user_001"
SESSION_ID = "support_session_001"

# Set your API key for the model you intend to use.
# Make sure GOOGLE_API_KEY is set in your environment variables.
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# -----------------------------
# LLM Model
# -----------------------------
# We'll use a consistent model for both agents for simplicity
AGENT_MODEL = LiteLlm("gemini/gemini-2.5-pro")

# -----------------------------
# Initialize and Wrap CrewAI Tool for ADK
# -----------------------------

# 1. Instantiate the CrewAI tool as you normally would.
# The URL from the notebook is used here.
docs_scrape_tool_instance = ScrapeWebsiteTool(
    website_url="https://naqshar.com/"
)

# 2. Create an async wrapper function to bridge CrewAI tool with ADK.
async def scrape_website_async(website_url: str) -> str:
    """
    Async wrapper for the CrewAI ScrapeWebsiteTool.
    This allows the synchronous tool to be used by the async ADK runner.
    """
    print(f"📄 Scraping website: {website_url}")
    try:
        # We call the tool's run method here.
        result = docs_scrape_tool_instance._run(website_url)
        # Truncate for token efficiency
        return str(result)[:4000] + "... [truncated]"
    except Exception as e:
        return f"Error scraping website: {str(e)}"

# 3. Create a Google ADK FunctionTool from the async wrapper.
scrape_tool = FunctionTool(func=scrape_website_async)


# =============================================================================
# AGENT 1: SUPPORT REPRESENTATIVE (ADK Conversion)
# =============================================================================
support_agent = LlmAgent(
    name="support_agent",
    model=AGENT_MODEL,
    description="A friendly and helpful senior support representative.",

    # The role, goal, backstory, and task from CrewAI are combined here.
    instruction="""
    ## YOUR ROLE
    You are a Senior Support Representative at the company.
    Your primary goal is to be the most friendly and helpful support representative on your team.

    ## CURRENT CONTEXT
    You are handling an inquiry from customer, a very important client.
    You must provide the best support possible support.
    Your answers must be complete, accurate, and make no assumptions.

    ## YOUR TASK
    You must address the following customer inquiry as outlined in prompt.
    
    ## AVAILABLE TOOLS
    - **scrape_website_async**: You can use this tool to read the content of the pre-configured documentation or company website to find relevant information.

    ## EXPECTED OUTPUT
    Produce a detailed, informative response that addresses all aspects of the customer's question.
    Your response must include references to any sources you used.
    Maintain a helpful, friendly, and professional tone throughout. Ensure no questions are left unanswered.
    """,
    tools=[scrape_tool] # This agent gets the tool, just like the task in CrewAI
)

# =============================================================================
# AGENT 2: SUPPORT QUALITY ASSURANCE (ADK Conversion)
# =============================================================================
qa_agent = LlmAgent(
    name="qa_agent",
    model=AGENT_MODEL,
    description="A specialist who ensures the quality and accuracy of support responses.",

    # The role, goal, backstory, and task for the QA agent are combined here.
    instruction="""
    ## YOUR ROLE
    You are a Support Quality Assurance Specialist at company.
    Your goal is to get recognition for providing the best support quality assurance in your team.

    ## YOUR TASK
    Review the response drafted by the Senior Support Representative for the inquiry from the customer.
    Your task is to ensure the response meets the highest quality standards.

    ## REVIEW CRITERIA
    1.  **Comprehensiveness**: Does the response fully address all parts of the customer's original inquiry?
    2.  **Accuracy**: Is the information provided correct and well-supported?
    3.  **Clarity**: Is the response easy to understand?
    4.  **Tone**: Is the tone helpful, friendly, and professional? Our company culture is chill and cool, so avoid being overly formal.
    5.  **References**: Are the sources used to find the information properly referenced?

    ## EXPECTED OUTPUT
    Provide the final, polished, and detailed response that is ready to be sent directly to the customer.
    This response should incorporate all necessary feedback and improvements to be a perfect answer.
    """,
    # This agent does not need tools, as its job is to review the prior agent's work.
    tools=[]
)


# =============================================================================
# SEQUENTIAL WORKFLOW (ADK's "Crew")
# =============================================================================
customer_support_crew = SequentialAgent(
    name='CustomerSupportCrew',
    description="A two-step crew for handling and quality-checking customer support inquiries.",
    sub_agents=[
        support_agent,
        qa_agent
    ],
)

# =============================================================================
# MAIN EXECUTION FUNCTION (ADK's "kickoff")
# =============================================================================
async def run_support_inquiry(inputs: dict):
    """
    Runs the customer support crew using the Google ADK Runner.
    """
    session_service = InMemorySessionService()
    print("🚀 Starting Customer Support Crew...")

    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=customer_support_crew, app_name=APP_NAME, session_service=session_service)

    # Format the initial prompt using the input dictionary
    query = f"""
    A new customer inquiry has arrived.

    - Customer: {inputs['customer']}
    - Contact Person: {inputs['person']}
    - Inquiry: {inputs['inquiry']}

    Please process this request through the standard support and QA workflow.
    """

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response = None

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            print(f"\n{'='*80}")
            print(f"Agent: {event.author}")
            print(f"{'='*80}")
            
            if event.is_final_response and event.author == "qa_agent":
                final_response = event.content.parts[0].text
                break # Stop after the final agent has responded

    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        return None

    print("\n\n🎉 Customer Support Crew Finished!")
    return final_response

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # The same inputs from the Jupyter Notebook
    inquiry_inputs = {
        "customer": "TensorDot Solution",
        "person": "Tariq Jamil",
        "inquiry": "I need help with arranging an internship in building design at Naqshar.com."
    }

    # Run the asynchronous function
    final_result = asyncio.run(run_support_inquiry(inquiry_inputs))

    if final_result:
        print("\n\n✅ Final QA-Approved Response:")
        print("---")
        print(final_result)