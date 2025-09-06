# -*- coding: utf-8 -*-
"""
Google ADK Conversion of a LangGraph Web Research Multi-Agent System.

Version 2: Refactored to use the native `LangchainTool` wrapper from the ADK
for a more direct and cleaner integration of LangChain tools.

This script replicates a multi-agent workflow for conducting web research.

Workflow:
1.  **Tavily Search Agent**: Takes a user's research query and uses the
    Tavily search tool to find a list of relevant, high-quality URLs.
2.  **Research Writer Agent**: Receives the list of URLs, scrapes their
    content using a web research tool, and then writes a comprehensive,
    long-form article in markdown format based on the gathered information.
"""

import os
import uuid
import asyncio
from pathlib import Path

# --- LangChain Tool Imports ---
# We will import the actual LangChain tools directly.
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.web import research # The custom tool from web.py

# --- ADK Imports ---
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.tools import FunctionTool
# Import the specialized wrapper for LangChain tools
from google.adk.tools.langchain_tool import LangchainTool

# --- Prompt Imports ---
from research_prompts import RESEARCHER_SYSTEM_PROMPT, TAVILY_AGENT_SYSTEM_PROMPT

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------
# Constants and API Key Checks
# -----------------------------
APP_NAME = "web_research_app_v2"
USER_ID = "research_user_002"
SESSION_ID = "research_session_002"
OUTPUT_DIRECTORY = Path("./output")
OUTPUT_DIRECTORY.mkdir(exist_ok=True)

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY environment variable not set.")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# -----------------------------
# LLM Model
# -----------------------------
AGENT_MODEL = LiteLlm("gemini/gemini-2.5-flash")

# =============================================================================
# TOOL INTEGRATION USING LangchainTool WRAPPER
# =============================================================================

# 1. Instantiate the standard LangChain tool.
tavily_tool_instance = TavilySearchResults(max_results=7)

# 2. Wrap the LangChain tool instance with ADK's LangchainTool.
#    No manual async wrapper is needed.
tavily_search_tool_adk = LangchainTool(
    tool=tavily_tool_instance
)

web_research_tool_adk = FunctionTool(
    func=research
)


# =============================================================================
# AGENT 1: TAVILY SEARCH AGENT
# =============================================================================
tavily_search_agent = LlmAgent(
    name="Tavily_Search_Agent",
    model=AGENT_MODEL,
    description="Uses Tavily to find relevant URLs for a given research topic.",
    instruction=TAVILY_AGENT_SYSTEM_PROMPT,
    # Use the wrapped ADK tool
    tools=[tavily_search_tool_adk],
)

# =============================================================================
# AGENT 2: RESEARCH WRITER AGENT
# =============================================================================
research_writer_agent = LlmAgent(
    name="Research_Writer_Agent",
    model=AGENT_MODEL,
    description="Reads content from URLs and writes a long-form article.",
    instruction=RESEARCHER_SYSTEM_PROMPT,
    # Use the wrapped ADK tool
    tools=[web_research_tool_adk],
)

# =============================================================================
# SEQUENTIAL WORKFLOW (The ADK "Graph")
# =============================================================================
web_research_system = SequentialAgent(
    name='WebResearchSystem',
    description="A sequential agent system that first finds sources and then writes a research article.",
    sub_agents=[
        tavily_search_agent,
        research_writer_agent,
    ],
)

# =============================================================================
# MAIN EXECUTION FUNCTION (No changes needed here)
# =============================================================================
async def run_web_research(query: str):
    """
    Runs the full web research workflow using the Google ADK Runner.
    """
    session_service = InMemorySessionService()
    print("🚀 Starting Web Research System (v2 with LangchainTool)...")

    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=web_research_system, app_name=APP_NAME, session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_article = None

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            print(f"--- Node: {event.author} ---")
            
            if event.is_final_response() and event.author == "Research_Writer_Agent":
                final_article = event.content.parts[0].text
                break

    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        return None

    print("\n🎉 Web Research System Finished!")
    return final_article

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    research_query = "Write a long-form LinkedIn newsletter in a professional tone on 'AI usage in Aviation'. Focus on Operations, Maintenance, and Safety management. Mention key companies using AI in aircraft operations and include verifiable source links."

    final_result = asyncio.run(run_web_research(research_query))

    if final_result:
        try:
            filename = f"{OUTPUT_DIRECTORY}/{uuid.uuid4()}.md"
            with open(filename, "w", encoding="utf-8") as file:
                file.write(final_result)
            print(f"\n✅ Research article successfully saved to: {filename}")
            print("\n--- Final Article Preview ---")
            print(final_result[:1000].strip() + "\n...")
            print("-------------------------")
        except Exception as e:
            print(f"❌ Error saving the final article: {e}")