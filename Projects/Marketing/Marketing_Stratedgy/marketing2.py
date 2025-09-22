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
import markdown
import pdfkit

# Import prompts and models from separate file
from prompts import (
    Copy,
    COMPANY_RESEARCH_ANALYST_INSTRUCTION,
    COMPETITIVE_INTELLIGENCE_SPECIALIST_INSTRUCTION,
    STRATEGIC_GROWTH_ADVISOR_INSTRUCTION,
    BUSINESS_DEVELOPMENT_CONSULTANT_INSTRUCTION,
    QUALITY_CONTROL_SPECIALIST_INSTRUCTION,
    EXECUTIVE_REPORT_SYNTHESIZER_INSTRUCTION
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
# LLM Model - Using single working model from code1
# -----------------------------
AGENT_MODEL = LiteLlm("gemini/gemini-2.0-flash")

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
        if 'organic' not in response_json or not response_json['organic']: 
            return "No organic results found."
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
        return str(result)[:8000]  # Limit content length
    except Exception as e:
        return f"An error occurred during scraping: {e}"

# -----------------------------
# Create Individual Tool Instances
# -----------------------------
search_tool = FunctionTool(func=search_internet)
scrape_tool = FunctionTool(func=scrape_website)

# =============================================================================
# ENHANCED AGENTS USING IMPORTED INSTRUCTIONS
# =============================================================================

# AGENT 1: Company Research Analyst
company_research_analyst = LlmAgent(
    name="company_research_analyst",
    model=AGENT_MODEL,
    description="Senior research analyst conducting comprehensive company intelligence with zero tolerance for incomplete data.",
    instruction=COMPANY_RESEARCH_ANALYST_INSTRUCTION,
    tools=[search_tool, scrape_tool],
    output_key="company_research"
)

# AGENT 2: Competitive Intelligence Specialist
competitive_intelligence_specialist = LlmAgent(
    name="competitive_intelligence_specialist",
    model=AGENT_MODEL,
    description="Expert competitive analyst with zero tolerance for incomplete competitor analysis.",
    instruction=COMPETITIVE_INTELLIGENCE_SPECIALIST_INSTRUCTION,
    tools=[search_tool, scrape_tool],
    output_key="competitive_intelligence"
)

# AGENT 3: Strategic Growth Advisor
strategic_growth_advisor = LlmAgent(
    name="strategic_growth_advisor",
    model=AGENT_MODEL,
    description="Senior strategy consultant developing specific, actionable growth strategies with no generic recommendations.",
    instruction=STRATEGIC_GROWTH_ADVISOR_INSTRUCTION,
    tools=[search_tool, scrape_tool],
    output_key="growth_strategy"
)

# AGENT 4: Business Development Consultant
business_development_consultant = LlmAgent(
    name="business_development_consultant",
    model=AGENT_MODEL,
    description="Senior business consultant providing specific, actionable business opportunities with detailed financial analysis.",
    instruction=BUSINESS_DEVELOPMENT_CONSULTANT_INSTRUCTION,
    tools=[search_tool, scrape_tool],
    output_key="business_development"
)

# AGENT 5: Quality Control Specialist
quality_control_specialist = LlmAgent(
    name="quality_control_specialist",
    model=AGENT_MODEL,
    description="Quality control specialist that identifies and fills gaps in previous outputs before final synthesis.",
    instruction=QUALITY_CONTROL_SPECIALIST_INSTRUCTION,
    tools=[search_tool, scrape_tool],
    output_key="enhanced_outputs"
)

# AGENT 6: Executive Report Synthesizer
executive_report_synthesizer = LlmAgent(
    name="executive_report_synthesizer",
    model=AGENT_MODEL,
    description="Executive consultant creating flawless, complete company reports with zero tolerance for gaps or placeholders.",
    instruction=EXECUTIVE_REPORT_SYNTHESIZER_INSTRUCTION,
    tools=[],  # Synthesis only - all research completed by previous agents
    output_schema=Copy,
    output_key="comprehensive_company_report",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# =============================================================================
# ENHANCED SEQUENTIAL WORKFLOW
# =============================================================================
root_agent = SequentialAgent(
    name='ComprehensiveCompanyReportAgent',
    description="""
    Execute enhanced company analysis workflow with quality control:
    1. Company Research Analyst - Complete foundational intelligence (no placeholders)
    2. Competitive Intelligence Specialist - Thorough market and competitor analysis
    3. Strategic Growth Advisor - Specific growth strategies with metrics
    4. Business Development Consultant - Detailed business opportunities with ROI
    5. Quality Control Specialist - Gap identification and additional research
    6. Executive Report Synthesizer - Professional, complete final report
    
    This enhanced workflow ensures zero placeholders, complete information,
    and professional-quality reports suitable for executive presentation.
    """,
    sub_agents=[
        company_research_analyst,
        competitive_intelligence_specialist,
        strategic_growth_advisor,
        business_development_consultant,
        quality_control_specialist,
        executive_report_synthesizer
    ],
)

# -----------------------------
# File Saving Function (From code1 - working)
# -----------------------------
def save_to_file(content: str, filename: str):
    try:
        data = json.loads(content)
        # Extract the "report" content
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

# -----------------------------
# Main Workflow Function
# -----------------------------
async def run_workflow(query: str):
    session_service = InMemorySessionService()
    print("🎨 Running Enhanced Marketing Workflow...")
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    creative_campaigns = "Content creation failed."
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            print(f"\n{event.author}\n{'*'*80}")
            if event.is_final_response() and event.author == "executive_report_synthesizer":
                creative_campaigns = event.content.parts[0].text
                print("✅ Enhanced Report Complete\n")
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