from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools.youtube.search import YouTubeSearchTool
from google.genai import types

session_service = InMemorySessionService()

APP_NAME = "Youtube_app"
USER_ID = "user_001"
SESSION_ID = "youtube_videos"

# Instantiate the tool
langchain_yt_tool = YouTubeSearchTool()

# Wrap the tool in the LangchainTool class from ADK
adk_yt_tool = LangchainTool(
    tool=langchain_yt_tool,
)

root_agent = LlmAgent(
    name="youtube_search_agent",
    model="gemini-1.5-flash",  # Replace with the actual model name
    instruction="""
    Ask customer to provide singer name, and the number of videos to search.
    """,
    description="Help customer to search for a video on Youtube.",
    tools=[adk_yt_tool],
    output_key="youtube_search_output",
)

print("******")
# Session and Runner
async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return session, runner


# Agent Interaction
async def call_agent_async(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session, runner = await setup_session_and_runner()
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

# Note: In Colab, you can directly use 'await' at the top level.
# If running this code as a standalone Python script, you'll need to use asyncio.run() or manage the event loop.
if __name__ == "__main__":
    import asyncio
    prompt = "Panaversity,10 videos?"
    asyncio.run(call_agent_async(prompt))