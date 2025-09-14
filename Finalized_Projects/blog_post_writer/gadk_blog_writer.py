# -*- coding: utf-8 -*-
"""
Running an advanced Google ADK multi-agent system for professional blog writing.

This script orchestrates a three-step content creation pipeline. It uses a more
narrative and professional prompting style, moving away from the rigid
'Role/Goal' format, while retaining full functionality, including web search for
sources and image search for embedding visuals.

The workflow consists of three specialized agents:
1.  **Content Planner**: Creating a detailed outline and finding source URLs.
2.  **Content Writer**: Writing the article, embedding images and sources.
3.  **Editor**: Reviewing and refining the final blog post for quality.
"""

import os
import asyncio
import json
from tavily import TavilyClient

# --- Importing ADK Components ---
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.genai import types
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# --- Setting up Environment and Constants ---
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Missing TAVILY_API_KEY for web and image search.")

APP_NAME = "blog_writer_app_pro"
USER_ID = "author_user_004"
SESSION_ID = "blog_session_004"
MODEL_NAME = "gemini/gemini-2.5-pro"

# --- Initializing LLM Model and API Clients ---
AGENT_MODEL = LiteLlm(MODEL_NAME)
TAVILY_CLIENT = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# =============================================================================
# DEFINING THE TOOLS
# =============================================================================

async def web_search_async(query: str) -> str:
    """
    Finding relevant web articles for a query and returning them as a
    formatted string with titles and URLs.
    """
    print(f"🔎 Searching the web for: '{query}'")
    try:
        results = TAVILY_CLIENT.search(query=query, search_depth="advanced")
        output = "\n".join([f"- [{res['title']}]({res['url']})" for res in results.get("results", [])[:4]])
        return output if output else "No relevant articles found."
    except Exception as e:
        return f"Error during web search: {e}"

web_search_tool = FunctionTool(func=web_search_async)

async def find_relevant_image_async(query: str) -> str:
    """
    Finding a relevant, real image for a given query and returning it as a
    markdown string.
    """
    print(f"🖼️ Searching for image with query: '{query}'")
    try:
        results = TAVILY_CLIENT.search(query=query, search_depth="advanced", include_images=True)
        if images := results.get("images", []):
            image_url = images[0]
            return f"![{query}]({image_url})"
        return "No relevant image found."
    except Exception as e:
        return f"Error during image search: {e}"

image_search_tool = FunctionTool(func=find_relevant_image_async)

# =============================================================================
# AGENT 1: CONTENT PLANNER
# =============================================================================
planner_agent = LlmAgent(
    name="Content_Planner",
    model=AGENT_MODEL,
    description="Planning engaging content and finding authoritative sources.",
    tools=[web_search_tool],
    instruction="""
    ### Persona and Objective
    You are an expert Content Planner. Your primary objective is to devise a comprehensive content strategy for a blog post about `{topic}` that is suitable for the style of `{style}`. Your plan will serve as the single source of truth for the Content Writer.

    ### Core Responsibilities
    1.  **Research Market Trends:** Prioritize the latest trends, key players, and noteworthy news related to `{topic}`.
    2.  **Find Authoritative Sources:** Use your web search tool to find 3-4 recent, high-quality articles or sources. These are critical for adding authenticity.
    3.  **Analyze the Audience:** Briefly describe the target audience, considering their interests and pain points.
    4.  **Develop Content Structure:** Create a detailed content outline, including an introduction, key talking points with sub-topics, and a compelling call to action.
    5.  **Identify SEO Keywords:** List relevant SEO keywords to be included in the article.

    ### Final Deliverable
    A comprehensive content plan document that includes the detailed outline, audience analysis, SEO keywords, and a list of 3-4 source URLs with their titles.
    """,
)

# =============================================================================
# AGENT 2: CONTENT WRITER
# =============================================================================
writer_agent = LlmAgent(
    name="Content_Writer",
    model=AGENT_MODEL,
    description="Writing an insightful article and embedding images and sources.",
    tools=[image_search_tool],
    instruction="""
    ### Persona and Objective
    You are a skilled Content Writer. Your mission is to transform the strategic plan provided by the Content Planner into a compelling, insightful, and well-researched blog post about `{topic}`.

    ### Core Responsibilities
    1.  **Write the Article:** Using the provided content plan, write a comprehensive blog post. Ensure the tone and style are appropriate for `{style}`.
    2.  **Integrate Sources:** Naturally weave the source links from the plan into the article where they support claims or provide further reading. Format them as markdown links, like `[Article Title](URL)`.
    3.  **Embed Imagery:** Use your image search tool to find and embed one or two relevant, high-quality images. Place them in appropriate sections to add visual interest.
    4.  **Incorporate Keywords:** Seamlessly integrate the provided SEO keywords throughout the text.
    5.  **Acknowledge Opinions:** When making statements that are your opinion rather than objective facts, clearly state this.

    ### Final Deliverable
    A well-written and complete blog post in markdown format. It must be at least 800 words long and include 1-2 embedded images and 3-4 embedded source links.
    """,
)

# =============================================================================
# AGENT 3: EDITOR
# =============================================================================
editor_agent = LlmAgent(
    name="Editor",
    model=AGENT_MODEL,
    description="Editing a blog post for style, grammar, and completeness.",
    instruction="""
    ### Persona and Objective
    You are a meticulous Editor. Your task is to perform the final quality assurance check on the blog post drafted by the Content Writer. You are the final gatekeeper before publication.

    ### Quality Assurance Checklist
    1.  **Tone and Style Alignment:** Does the post's tone perfectly match the `{style}`? Is it professional yet engaging?
    2.  **Grammar and Clarity:** Is the article free of all grammatical errors and typos? Is the language clear, concise, and easy to read?
    3.  **Completeness Check:** Verify that 3-4 source links and 1-2 images are present and correctly formatted in markdown.
    4.  **Journalistic Integrity:** Ensure any opinions are clearly stated as such, maintaining a balanced viewpoint.

    ### Final Deliverable
    A final, polished, and publish-ready blog post in markdown format.
    """,
)

# =============================================================================
# DEFINING THE SEQUENTIAL WORKFLOW
# =============================================================================
blog_writing_crew = SequentialAgent(
    name='BlogWritingCrew',
    description="A three-step crew for planning, writing, and editing a blog post.",
    sub_agents=[
        planner_agent,
        writer_agent,
        editor_agent,
    ],
)

# =============================================================================
# RUNNING THE MAIN EXECUTION FUNCTION
# =============================================================================
async def run_blog_creation(inputs: dict):
    """
    Running the blog creation workflow using the Google ADK Runner.
    """
    session_service = InMemorySessionService()
    print("🚀 Starting Professional Blog Writing Workflow...")

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"topic": inputs['topic'], "style": inputs['style']}
    )
    runner = Runner(agent=blog_writing_crew, app_name=APP_NAME, session_service=session_service)

    initial_message = types.Content(role='user', parts=[types.Part(text="Start the blog creation process.")])
    final_blog_post = None

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=initial_message):
            print(f"--- Agent in control: {event.author} ---")
            if event.is_final_response() and event.author == "Editor":
                final_blog_post = event.content.parts[0].text
                break
    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        return None

    print("\n🎉 Professional Blog Writing Workflow Finished!")
    return final_blog_post

# =============================================================================
# EXECUTING THE SCRIPT
# =============================================================================
if __name__ == "__main__":
    blog_inputs = {
        "style": "a professional but engaging tone suitable for www.medium.com",
        "topic": "The future of AI in commercial aviation, focusing on predictive maintenance and flight operations."
    }

    final_result = asyncio.run(run_blog_creation(blog_inputs))

    if final_result:
        print("\n✅ Final, Edited Blog Post:")
        print("---------------------------------")
        print(final_result)
        print("---------------------------------")

        output_filename = "final_blog_post_professional.md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_result)
        print(f"\n📄 Blog post saved to '{output_filename}'")