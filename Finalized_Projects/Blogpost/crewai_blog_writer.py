# -*- coding: utf-8 -*-
"""
Blog Writer Script

This script automates the creation of a blog post using a crew of AI agents (Planner, Writer, Editor)
to produce content in the style of a specified publication (e.g., Aviation Week) on a given topic.
"""

# =====================================================================
# Imports and Configuration
# =====================================================================
import os
from typing import Dict
from crewai import Agent, Task, Crew
from crewai import LLM as LiteLLM
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize LLM
def setup_llm() -> LiteLLM:
    """Set up the LiteLLM instance for Groq's LLaMA model.

    Returns:
        Configured LiteLLM instance.

    Raises:
        ValueError: If the GROQ_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        raise ValueError("GROQ_API_KEY environment variable not set")

    logger.info("Initializing LiteLLM with Groq LLaMA3-70b-8192 model")
    return LiteLLM(
        model="gemini/gemini-2.0-flash",
        #base_url="https://api.groq.com/openai/v1",
        #api_key=api_key
    )

# =====================================================================
# Agent Definitions
# =====================================================================
def create_planner_agent(llm: LiteLLM, style: str) -> Agent:
    """Create a Content Planner Agent for blog post planning.

    Args:
        llm: LiteLLM instance for the agent.
        style: Publication style for the blog post (e.g., 'www.aviationweek.com').

    Returns:
        Configured Planner Agent instance.
    """
    logger.info("Creating Content Planner Agent")
    return Agent(
        role="Content Planner",
        goal="Plan engaging and factually accurate content on {topic}",
        backstory=(
            f"You're working on planning a blog article about the topic: {{topic}} in {style}. "
            "You collect information that helps the audience learn something and make informed decisions. "
            "You have to prepare a detailed outline and the relevant topics and sub-topics that has to be a part of the blogpost. "
            "Your work is the basis for the Content Writer to write an article on this topic."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

def create_writer_agent(llm: LiteLLM, style: str) -> Agent:
    """Create a Content Writer Agent for drafting the blog post.

    Args:
        llm: LiteLLM instance for the agent.
        style: Publication style for the blog post (e.g., 'www.aviationweek.com').

    Returns:
        Configured Writer Agent instance.
    """
    logger.info("Creating Content Writer Agent")
    return Agent(
        role="Content Writer",
        goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
        backstory=(
            f"You're working on writing a new opinion piece about the topic: {{topic}} in {style}. "
            "You base your writing on the work of the Content Planner, who provides an outline and relevant context about the topic. "
            "You follow the main objectives and direction of the outline, as provided by the Content Planner. "
            "You also provide objective and impartial insights and back them up with information provided by the Content Planner. "
            "You acknowledge in your opinion piece when your statements are opinions as opposed to objective statements."
        ),
        allow_delegation=False,
        llm=llm,
        verbose=True
    )

def create_editor_agent(llm: LiteLLM, style: str) -> Agent:
    """Create an Editor Agent for refining the blog post.

    Args:
        llm: LiteLLM instance for the agent.
        style: Publication style for the blog post (e.g., 'www.aviationweek.com').

    Returns:
        Configured Editor Agent instance.
    """
    logger.info("Creating Editor Agent")
    return Agent(
        role="Editor",
        goal=f"Edit a given blog post to align with the writing style of the organization {style}.",
        backstory=(
            "You are an editor who receives a blog post from the Content Writer. "
            "Your goal is to review the blog post to ensure that it follows journalistic best practices, "
            "provides balanced viewpoints when providing opinions or assertions, "
            "and also avoids major controversial topics or opinions when possible."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

# =====================================================================
# Task Definitions
# =====================================================================
def create_plan_task(planner: Agent) -> Task:
    """Create a task for the Content Planner Agent.

    Args:
        planner: Content Planner Agent instance.

    Returns:
        Configured Task instance for planning.
    """
    logger.info("Creating Content Planning Task")
    return Task(
        description=(
            "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
            "2. Identify the target audience, considering their interests and pain points.\n"
            "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
            "4. Include SEO keywords and relevant data or sources."
        ),
        expected_output=(
            "A comprehensive content plan document with an outline, audience analysis, "
            "SEO keywords, and resources."
        ),
        agent=planner
    )

def create_write_task(writer: Agent) -> Task:
    """Create a task for the Content Writer Agent.

    Args:
        writer: Content Writer Agent instance.

    Returns:
        Configured Task instance for writing.
    """
    logger.info("Creating Content Writing Task")
    return Task(
        description=(
            "1. Use the content plan to craft a compelling blog post on {topic}.\n"
            "2. Incorporate SEO keywords naturally.\n"
            "3. Sections/Subtitles are properly named in an engaging manner.\n"
            "4. Ensure the post is structured with an engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
            "5. Proofread for grammatical errors and alignment with the brand's voice.\n"
            "6. Always insert specific company names and URL links when making reference to some plans, figures, or data."
        ),
        expected_output=(
            "A well-written blog post in markdown format, ready for publication, "
            "each section should have 2 or 3 paragraphs. "
            "The length of article should be greater than 800 words"
        ),
        agent=writer
    )

def create_edit_task(editor: Agent) -> Task:
    """Create a task for the Editor Agent.

    Args:
        editor: Editor Agent instance.

    Returns:
        Configured Task instance for editing.
    """
    logger.info("Creating Editing Task")
    return Task(
        description=(
            "Proofread the given blog post for grammatical errors and "
            "alignment with the brand's voice."
        ),
        expected_output=(
            "A well-written blog post in markdown format, ready for publication, "
            "each section should have 2 or 3 paragraphs."
        ),
        agent=editor
    )

# =====================================================================
# Crew Setup and Execution
# =====================================================================
def main():
    """Main function to execute the blog writing process."""
    logger.info("Starting blog writing process")

    # Initialize LLM
    try:
        llm = setup_llm()
    except ValueError as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

    # Set publication style
    style = "'www.aviationweek.com'"
    logger.info(f"Using publication style: {style}")

    # Create agents
    planner = create_planner_agent(llm, style)
    writer = create_writer_agent(llm, style)
    editor = create_editor_agent(llm, style)

    # Create tasks
    plan_task = create_plan_task(planner)
    write_task = create_write_task(writer)
    edit_task = create_edit_task(editor)

    # Initialize crew
    logger.info("Initializing CrewAI with agents and tasks")
    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan_task, write_task, edit_task],
        verbose=True
    )

    # Define inputs and execute
    inputs = {
        "style": style,
        "topic": "Exploring the potential of AI usage in Aircraft Lease Return"
    }
    logger.info(f"Executing crew with topic: {inputs['topic']}")
    try:
        result = crew.kickoff(inputs=inputs)
        logger.info("Blog writing process completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during crew execution: {e}")
        raise

# =====================================================================
# Entry Point
# =====================================================================
if __name__ == "__main__":
    main()

    # Optional: Convert markdown output to docx (uncomment to use)
    import subprocess
    subprocess.run(["pandoc", "am.txt", "-o", "am.docx"])