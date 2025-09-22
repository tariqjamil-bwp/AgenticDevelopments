# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

import os
# Disable telemetry to prevent harmless exit errors
os.environ["OTEL_SDK_DISABLED"] = "true"

import asyncio
import json
import httpx
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from crewai_tools import ScrapeWebsiteTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import markdown
import pdfkit

from prompts import (
    LEAD_MARKET_ANALYST_INSTRUCTION,
    CHIEF_MARKETING_STRATEGIST_INSTRUCTION,
    CREATIVE_CONTENT_CREATOR_INSTRUCTION,
    CHIEF_CREATIVE_DIRECTOR_INSTRUCTION,
)

project_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_path)
# -----------------------------
# Constants and API Key Checks
# -----------------------------
APP_NAME = "marketing_workflow_app"
USER_ID = "user_001"
SESSION_ID = "marketing_session_001"

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY environment variable not set.")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# -----------------------------
# LLM Model
# -----------------------------
AGENT_MODEL = LiteLlm("gemini/gemini-2.5-flash")

# -----------------------------
# Define Asynchronous Tool Functions
# -----------------------------
async def search_internet(query: str) -> str:
    """Useful to search the internet about a given topic and return relevant results."""
    print(f"--- Using Tool: search_internet with query: '{query}' ---")
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {'X-API-KEY': os.environ['SERPER_API_KEY'], 'Content-Type': 'application/json'}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, content=payload)
        response.raise_for_status()
        response_json = response.json()
        if 'organic' not in response_json or not response_json['organic']: return "No organic results found."
        results = response_json['organic'][:4]
        string = [f"Title: {r.get('title', 'N/A')}\nLink: {r.get('link', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}\n---" for r in results]
        return '\n'.join(string)
    except Exception as e:
        return f"An error occurred during search: {e}"

async def scrape_website(website_url: str) -> str:
    """Scrapes the content of a given website URL."""
    print(f"--- Using Tool: scrape_website with URL: '{website_url}' ---")
    try:
        scraper_instance = ScrapeWebsiteTool()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: scraper_instance._run(website_url=website_url))
        return str(result)[:8000] # Limit content length
    except Exception as e:
        return f"An error occurred during scraping: {e}"

# -----------------------------
# Create Individual Tool Instances
# -----------------------------
search_tool = FunctionTool(func=search_internet)
scrape_tool = FunctionTool(func=scrape_website)

# -----------------------------
# Pydantic Model for Final Output
# -----------------------------
class Copy(BaseModel):
    """The final, client-ready marketing report."""
    report: str = Field(description="The full report in markdown format.")

lead_market_analyst = LlmAgent(
    name="lead_market_analyst",
    model=AGENT_MODEL,
    description= "A Lead Market Analyst who conducts in-depth analysis of products and competitors.",
    instruction= LEAD_MARKET_ANALYST_INSTRUCTION,
    tools=[search_tool, scrape_tool],
    output_key="market_analysis",
)

chief_marketing_strategist = LlmAgent(
    name="chief_marketing_strategist",
    model=AGENT_MODEL,
    description="A Chief Marketing Strategist who formulates innovative marketing strategies.",
    instruction= CHIEF_MARKETING_STRATEGIST_INSTRUCTION,
    tools=[search_tool, scrape_tool],
    output_key="marketing_strategy"
)

creative_content_creator = LlmAgent(
    name="creative_content_creator",
    model=AGENT_MODEL,
    description="A Creative Content Creator who develops compelling and innovative content.",
    instruction= CREATIVE_CONTENT_CREATOR_INSTRUCTION,
    tools=[search_tool],
    output_key="created_content"
)

chief_creative_director = LlmAgent(
    name="chief_creative_director",
    model=AGENT_MODEL,
    description="A Chief Creative Director who finalizes all marketing content.",
    instruction= CHIEF_CREATIVE_DIRECTOR_INSTRUCTION,
    output_schema=Copy,
    output_key="final_report",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

root_agent = SequentialAgent(
    name='MarketingWorkflowAgent',
    description="Run the workflow Market Analyst, then Chief Marketing Strategist, then Creative Content Creator, and finally Chief Creative Director.",
    sub_agents=[lead_market_analyst, chief_marketing_strategist, creative_content_creator, chief_creative_director],
    )

def save_to_file(content: str, filename: str):
    try:
        data = json.loads(content)
        # Extract the "report" content (equivalent to "text")
        text = data["report"]
        with open(filename, "w") as f:
            f.write(text)

        if text:
            html_content = markdown.markdown(text)
            pdfkit.from_string(html_content, "marketing_report.pdf")
            return "Content created successfully."
        else:
            return "Content creation failed."

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return "Content creation failed."
    except Exception as e:
        print(f"Error: {e}")
        return f"Content creation failed with error: {e}."

#***************************************************
async def run_workflow(query: str):
    session_service = InMemorySessionService()
    print("🎨 Running Creative Content...")
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    creative_campaigns = "Content creation failed."
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            print(f"\n{event.author}\n{'*'*80}")            
            if event.is_final_response() and event.author == "chief_creative_director":
                creative_campaigns = event.content.parts[0].text
                print("✅ Creative Content Complete\n")
                print(creative_campaigns)
                save_to_file(creative_campaigns, "marketing_report.md")
                break
    except GeneratorExit:
        print("Generator cleanup handled.")
    except Exception as e:
        print(f"❌ Error: {e}")
            
# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    
    # query = (
    # "Customer Domain: https://piac.com.pk/\n\n"
    # "Generate a comprehensive market analysis and strategy report for Pakistan International Airlines (PIA). Include:\n\n"
    # "Market Analysis: Industry overview, key trends, competitive landscape, SWOT analysis, market size/share, customer segments, and economic/regulatory factors.\n"
    # "Strategy Recommendations: Growth strategies, operational improvements, digital transformation, partnerships, financial turnaround plans, and measurable KPIs."
    # "Use up-to-date data from reliable sources. Structure the report in markdown with clear sections and subsections."
    # )

    # query = ("Customer Domain: https://aibytec.com/\n\n"
    # "Develop a comprehensive marketing strategy for AibyTec, an emerging AI solutions provider specializing in customized AI-driven tools like intelligent chatbots, AI avatars, machine learning models, and deep learning applications for industries such as healthcare and corporate automation. Target tech-savvy decision-makers (e.g., CTOs, CIOs, CEOs) in the local market, emphasizing ROI, efficiency, and innovation."
    # "Include:\n"
    # "- **Market Analysis**: Industry overview, trends in AI adoption, competitive landscape, SWOT analysis, target audience segments, market size/share, and local economic/regulatory factors.\n"
    # "- **Marketing Objectives**: Specific, measurable goals for brand awareness, lead generation, and client acquisition.\n"
    # "- **Strategy Components**: Positioning, value proposition, 4Ps (Product, Price, Place, Promotion), key messaging, channels (digital, content marketing, events, partnerships), and budget/timeline considerations.\n"
    # "- **Implementation Plan**: Tactics, KPIs for measurement, and recommendations for adaptation.\n"
    # "Use up-to-date data from reliable sources. Structure the report in markdown with clear sections and subsections.")
    
    query = "Customer Domain: https://www.naqshar.com/\n\n"
    "Generate a comprehensive report for Naqshar, an architecture design company specializing in sustainable residential and commercial projects. Include:\n\n"
    "- **Company Outlook**: Overview of current market position, service offerings, strengths, and competitive advantages\n"
    "- **Future Growth Ideas**: Expansion strategies, new service lines, technology adoption\n"
    "- **Marketing Campaigns**: 5 creative campaign ideas with competitor comparison\n"
    "- **Financial Analysis**: Use PKR currency (1 USD = 290 PKR)\n"
    "Structure the report in markdown with clear sections and subsections."
    
    try:
        asyncio.run(run_workflow(query))
    except KeyboardInterrupt:
        print("\n❌ Workflow interrupted by user")
    except Exception as e:
        print(f"❌ A fatal error occurred: {e}")