#!/usr/bin/env python3
"""
E-commerce Recommendation AI Agents with ADK + Vector Search

This module implements a multi-agent system for e-commerce generative recommendations:
- Research Agent: Uses Google Search to research user intents
- Shop Agent: Coordinates research and item finding using vector search

Original Author(s): Kaz Sato
Converted from Jupyter notebook to Python script (direct agent definition)
"""

import os
import logging
import asyncio
import requests
import json
from typing import Dict, List
from getpass import getpass

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search
from google.genai import types

if os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    del os.environ["GEMINI_API_KEY"]

# Configure logging
logging.getLogger("google.adk.runners").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

# Configuration
APP_NAME = "shop_concierge_app"
USER_ID = "user_1"
VECTOR_SEARCH_URL = "https://www.ac0.cloudadvocacyorg.joonix.net/api/query"

# Set up environment variables
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
    api_key = getpass("Enter your Gemini API Key: ")
    os.environ["GOOGLE_API_KEY"] = api_key


def call_vector_search(query: str, rows: int = None) -> dict:
    """Calls the Vector Search backend for querying."""
    headers = {'Content-Type': 'application/json'}
    payload = {
        "query": query,
        "rows": rows,
        "dataset_id": "mercari3m_mm",
        "use_dense": True,
        "use_sparse": True,
        "rrf_alpha": 0.5,
        "use_rerank": True,
    }

    try:
        response = requests.post(VECTOR_SEARCH_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling the API: {e}")
        return None


def find_shopping_items(queries: List[str]) -> List[dict]:
    """
    Find shopping items from the e-commerce site with the specified list of queries.

    Args:
        queries: the list of queries to run.
    
    Returns:
        List of items found in the e-commerce site.
    """
    items = []
    for query in queries:
        result = call_vector_search(query=query, rows=3)
        if result and "items" in result:
            items.extend(result["items"])

    print("-----")
    print(f"User queries: {queries}")
    print(f"Found: {len(items)} items")
    print("-----")

    return items


# Initialize session service
session_service = InMemorySessionService()

# Define Research Agent directly
research_agent = Agent(
    model='gemini-2.5-flash',
    name='research_agent',
    description='''
        A market researcher for an e-commerce site. Receives a search request
        from a user, and returns a list of 5 generated queries in English.
    ''',
    instruction='''
        Your role is a market researcher for an e-commerce site with millions of
        items.

        When you received a search request from a user, use Google Search tool to
        research on what kind of items people are purchasing for the user's intent.

        Then, generate 5 queries finding those items on the e-commerce site and
        return them.
    ''',
    tools=[google_search],
)

# Define Shop Agent directly
shop_agent = Agent(
    model='gemini-2.5-flash',
    name='shop_agent',
    description='A shopper\'s concierge for an e-commerce site',
    instruction='''
        Your role is a shopper's concierge for an e-commerce site with millions of
        items. Follow the following steps.

        When you received a search request from a user, pass it to `research_agent`
        tool, and receive 5 generated queries. Then, pass the list of queries to
        `find_shopping_items` to find items. When you received a list of items from
        the tool, answer to the user with item's name, description and the image url.
    ''',
    tools=[
        AgentTool(agent=research_agent),
        find_shopping_items,
    ],
)

# Define Simple Shop Agent (without research capability)
simple_shop_agent = Agent(
    model='gemini-2.5-flash',
    name='simple_shop_agent',
    description='Shop agent for an e-commerce site',
    instruction='''
        Your role is a shop search agent on an e-commerce site with millions of
        items. Your responsibility is to search items based on the queries you
        receive.

        To find items use `find_shopping_items` tool by passing a list of queries,
        and answer to the user with item's name, description and img_url
    ''',
    tools=[find_shopping_items],
)


async def test_agent(query: str, agent: Agent) -> str:
    """Sends a query to the agent and returns the final response."""
    print(f"\n>>> User Query: {query}")

    # Create a session
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    # Create a Runner
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = None
    # Iterate through events from run_async to find the final answer
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            break
    
    print(f"<<< Agent Response: {final_response_text}")
    return final_response_text


def test_vector_search():
    """Test the vector search functionality directly."""
    print("\nTesting Vector Search directly...")
    test_queries = ["Cups with dancing people", "Cups with dancing animals"]
    results = find_shopping_items(test_queries)
    
    print(f"Found {len(results)} items:")
    for i, item in enumerate(results[:3], 1):  # Show first 3 items
        print(f"{i}. {item['name']}")
        print(f"   Description: {item['description'][:100]}...")
        print(f"   URL: {item['url']}")


async def run_demo_tests():
    """Run predefined demo tests."""
    print("E-commerce Recommendation AI Agent System")
    print("="*60)
    
    # Test vector search first
    test_vector_search()
    
    # Test queries
    test_queries = [
        "Can you find birthday present for 10 years old son? He loves cars. Suggest an online buying link also.",
        "I need a gift for my sister who loves cooking",
        "Looking for workout equipment for home gym",
        "What are good tech gadgets for college students?",
    ]
    
    print("\nTesting the complete shop agent system:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}]")
        try:
            resp = await test_agent(query, shop_agent)
        except Exception as e:
            print(f"Error in test {i}: {e}")
        
        if resp:
            mode = "w+" if i == 1 else "a+"
            with open(f"shopping_suggestions.txt", mode) as f:
                f.write(resp)
        
        if i < len(test_queries):
            print("\n" + "-"*40)
    
    print("\n" + "="*60)
    print("TESTING RESEARCH AGENT ONLY")
    print("="*60)
    
    # Test research agent separately
    research_test_query = "birthday present for 10 years old boy who loves cars"
    print(f"\nTesting research agent with: {research_test_query}")
    try:
        await test_agent(research_test_query, research_agent)
    except Exception as e:
        print(f"Error in research agent test: {e}")


async def run_interactive_mode():
    """Run the system in interactive mode."""
    print("\n" + "="*60)
    print("INTERACTIVE E-COMMERCE RECOMMENDATION SYSTEM")
    print("="*60)
    print("Enter your shopping queries. Type 'quit' to exit.")
    print("Examples:")
    print("  - 'birthday gift for teenager who likes music'")
    print("  - 'home office equipment for remote work'")
    print("  - 'fitness gear for beginners'")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nYour query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query:
                await test_agent(query, shop_agent)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


async def test_simple_agent():
    """Test the simple shop agent without research functionality."""
    print("\n" + "="*60)
    print("TESTING SIMPLE SHOP AGENT (NO RESEARCH)")
    print("="*60)
    
    test_query = "Cups with dancing figures"
    print(f"Testing simple agent with: {test_query}")
    
    try:
        await test_agent(test_query, simple_shop_agent)
    except Exception as e:
        print(f"Error in simple agent test: {e}")


async def main():
    """Main function to demonstrate the shop agent system."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            await run_interactive_mode()
        elif sys.argv[1] == "--simple":
            await test_simple_agent()
        elif sys.argv[1] == "--vector-test":
            test_vector_search()
        else:
            print("Usage: python shop_agent.py [--interactive|--simple|--vector-test]")
    else:
        await run_demo_tests()


if __name__ == "__main__":
    asyncio.run(main())