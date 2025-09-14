# -*- coding: utf-8 -*-
"""
Google ADK Multi-Agent Note Formatter.

This script processes an input markdown note through a sequential multi-agent
workflow to produce a single, well-formatted markdown output file.

Workflow:
1.  **Topic Analyzer**: Reads the raw note and extracts key topics.
2.  **Paraphraser**: Rewrites the note, using the extracted topics as section
    headers for better organization.
3.  **Task Appender**: Takes the rewritten note and appends a final
    "Action Items" section, listing all actionable tasks found in the text.

The entire process is orchestrated using a Google ADK SequentialAgent,
demonstrating a clear, step-by-step document transformation pipeline.
"""

import os
import asyncio
import argparse
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

# -----------------------------
# Constants and API Key Checks
# -----------------------------
APP_NAME = "note_formatter_app"
USER_ID = "admin_user_001"
SESSION_ID = "note_format_session_001"

# This script can be adapted to any model provider supported by LiteLLM.
# For this example, we'll assume a Google API key is set.
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
# Change the current working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------
# LLM Model
# -----------------------------
AGENT_MODEL = LiteLlm("gemini/gemini-2.5-flash")

# =============================================================================
# AGENT 1: TOPIC ANALYZER
# =============================================================================
topic_analyzer_agent = LlmAgent(
    name="Topic_Analyzer",
    model=AGENT_MODEL,
    description="Analyzes a note to generate a list of important topics.",
    instruction="""You are a helpful AI assistant.
Your task is to analyze the provided note and generate a concise list of the main topics discussed. This list will be used by the next agent to structure a document.

"RULES":
- Output must be a simple, bulleted list of topics.
- Start each topic with a hyphen ('-').
- Extract only the most important topics.
- Do not reuse the exact phrasing from the note; summarize the concepts.
- Provide at least one topic, but no more than ten.
- Your output should ONLY be the list of topics, with no preamble or explanation.""",
)

# =============================================================================
# AGENT 2: PARAPHRASER
# =============================================================================
paraphraser_agent = LlmAgent(
    name="Paraphraser",
    model=AGENT_MODEL,
    description="Rewrites a note into a structured markdown document using a list of topics.",
    instruction="""You are an expert AI content editor.
You will receive two inputs from the previous steps: the original user's note and a list of topics.
Your job is to rewrite the original note into a well-structured markdown document.

"RULES":
- Your output must be in markdown format.
- Use each topic from the provided list as a main header in your output.
- Each header must start with `##`.
- Populate the content under each header using relevant information from the original note.
- Ensure all information from the original note is included under the appropriate header.
- If some information from the note does not fit any topic, create a final header named "## Additional Info" and place it there.
- Correct any spelling and grammatical mistakes you find in the original content.""",
)

# =============================================================================
# AGENT 3: TASK APPENDER (Generates Markdown, not YAML)
# =============================================================================
task_appender_agent = LlmAgent(
    name="Task_Appender",
    model=AGENT_MODEL,
    description="Identifies actionable tasks and appends them to the formatted note.",
    instruction="""You are a helpful AI personal assistant.
You will receive a formatted markdown note from the previous agent. Your final job is to identify any action items from the note and append them to the very end to create a complete, final document.

"RULES":
- Your output must be a single, complete markdown document.
- **First**, you must include the full, unaltered text of the formatted note you received from the previous agent.
- **Second**, after all the content from the previous agent, you must add a new header: `## Action Items`.
- Under the "Action Items" header, create a bulleted list of all the tasks or next steps you identified from the note.
- Each task in the list must start with a hyphen ('-').
- Each task should be a clear, concise, and actionable statement.
- If no specific tasks are found, write "No specific action items were identified." under the header.""",
)

# =============================================================================
# SEQUENTIAL WORKFLOW (The ADK "Group Chat")
# =============================================================================
note_formatting_system = SequentialAgent(
    name='NoteFormattingSystem',
    description="A multi-agent system to analyze, rewrite, and extract tasks from a note, producing a final markdown document.",
    sub_agents=[
        topic_analyzer_agent,
        paraphraser_agent,
        task_appender_agent,
    ],
)

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
async def run_note_formatting(note_content: str):
    """
    Runs the full note formatting workflow using the Google ADK Runner.
    """
    session_service = InMemorySessionService()
    print("🚀 Starting Note Formatting System...")

    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=note_formatting_system, app_name=APP_NAME, session_service=session_service)

    initial_prompt = f"Please process the following note:\n\n---\n{note_content}"
    content = types.Content(role='user', parts=[types.Part(text=initial_prompt)])
    final_markdown_output = None

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            print(f"[{event.author}] is working...")

            # The final, consolidated output comes from the last agent in the sequence.
            if event.is_final_response() and event.author == "Task_Appender":
                print(f"✅ Received final markdown document from {event.author}.")
                final_markdown_output = event.content.parts[0].text
                break

    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        return None

    print("\n🎉 Note Formatting System Finished!")
    return final_markdown_output

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a markdown note with a Google ADK multi-agent system.")
    parser.add_argument(
        "-i", "--input-file",
        default="notes.md",
        type=str,
        help="The path to the input markdown note file."
    )
    parser.add_argument(
        "-o", "--output-file",
        default="formatted_note.md",
        type=str,
        help="The path to save the final formatted markdown file."
    )
    args = parser.parse_args()

    try:
        with open(args.input_file, "r", encoding='utf-8') as f:
            note = f.read()
            print(f"📄 Loaded note from: {args.input_file}")

        # Run the asynchronous ADK workflow
        final_result = asyncio.run(run_note_formatting(note))

        if final_result:
            with open(args.output_file, "w", encoding='utf-8') as f:
                f.write(final_result)
            print(f"\n✅ Successfully generated and saved formatted note to: {args.output_file}")
            print("\n--- Document Preview ---")
            print(final_result)
            print("----------------------")

    except FileNotFoundError:
        print(f"❌ Error: The input file '{args.input_file}' was not found.")
    except Exception as e:
        print(f"❌ A fatal error occurred: {e}")