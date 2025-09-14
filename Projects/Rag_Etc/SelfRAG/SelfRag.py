# adk_self_rag_no_langchain.py

# =============================================================================
# 0. IMPORTS
# =============================================================================
# --- Standard Library Imports ---
import os
import asyncio
import json
from typing import List, Dict, Any

# --- Third-Party Package Imports ---
from dotenv import load_dotenv
import chromadb
import ollama
from openai import OpenAI
from tavily import TavilyClient

# --- Google ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.genai import types

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# =============================================================================
# 1. SETUP: CONFIGURATION AND CONSTANTS
# =============================================================================
# Loading environment variables from .env file
load_dotenv()
required_vars = ["TAVILY_API_KEY", "GEMINI_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Environment variable '{var}' is not set.")

# --- Model and Service Configuration ---
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
GEMINI_MODEL_ID = "gemini-1.5-flash"  # Using a more standard Gemini model name
CHROMA_DB_PATH = "local_chroma_db"
COLLECTION_NAME = "local_knowledge"

# --- ADK Configuration ---
APP_NAME = "self_rag_no_lc_app"
USER_ID = "researcher_003"
SESSION_ID = "self_rag_session_003"

# =============================================================================
# 2. INITIALIZING CLIENTS AND VECTORSTORE
# =============================================================================
print("🔧 Initializing clients and local ChromaDB store...")

# --- Initializing LLM Clients ---
# ADK Agent uses LiteLLM wrapper for its main reasoning loop
AGENT_LLM = LiteLlm(model=f"gemini/{GEMINI_MODEL_ID}", temperature=0.2)
# Tools use a direct client for specific tasks like JSON generation
LLM_CLIENT_FOR_TOOLS = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/",
    api_key=os.getenv("GEMINI_API_KEY")
)
# Tavily client for web search
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- Initializing Vector Store Client ---
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print("✅ Initialization complete.")
except Exception as e:
    print(f"❌ Error loading local ChromaDB store: {e}")
    print(f"Please run `setup_docs.py` first to create the '{CHROMA_DB_PATH}' directory.")
    exit()

# =============================================================================
# 3. DEFINING CORE LOGIC FUNCTIONS
# =============================================================================
async def llm_json_request(prompt: str, llm_client: OpenAI) -> Dict[str, Any]:
    """Helper function to make a request to an LLM and parse the JSON response."""
    try:
        # Note: Using asyncio.to_thread because the OpenAI v1+ client's async is
        # handled differently. This ensures non-blocking execution.
        chat_completion = await asyncio.to_thread(
            llm_client.chat.completions.create,
            messages=[{"role": "user", "content": prompt}],
            model=GEMINI_MODEL_ID,
            temperature=0,
            response_format={"type": "json_object"},
        )
        response_text = chat_completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        print(f"--- LLM JSON Request Failed: {e} ---")
        return {"score": "no"}  # Default to a safe failure mode

async def grade_relevance_async(question: str, document: str) -> Dict[str, Any]:
    """Core logic to grade a document's relevance using an LLM."""
    prompt = f"""You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. The goal is to filter out erroneous retrievals.
    Give a binary 'yes' or 'no' score. Provide the score as a JSON with a single key 'score'.
    Document: {document}
    Question: {question}"""
    return await llm_json_request(prompt, LLM_CLIENT_FOR_TOOLS)

async def grade_hallucination_async(answer: str, documents: List[str]) -> Dict[str, Any]:
    """Core logic to grade an answer for factual grounding."""
    context = "\n\n".join(documents)
    prompt = f"""You are a grader assessing whether an answer is grounded in a set of facts. Give a binary 'yes' or 'no' score. Provide the score as a JSON with a single key 'score'.
    Facts: {context}
    Answer: {answer}"""
    return await llm_json_request(prompt, LLM_CLIENT_FOR_TOOLS)

async def generate_answer_from_context_async(question: str, context: List[str]) -> str:
    """Core logic to generate a final answer from a given context."""
    full_context = "\n\n".join(context)
    prompt = f"""You are a helpful assistant. Use a neutral, professional tone. Use only the information in the context.
    Question: {question}
    Context: {full_context}
    Answer:"""
    chat_completion = await asyncio.to_thread(
        LLM_CLIENT_FOR_TOOLS.chat.completions.create,
        messages=[{"role": "user", "content": prompt}],
        model=GEMINI_MODEL_ID,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content

# =============================================================================
# 4. DEFINING AND INSTANTIATING ADK TOOLS
# =============================================================================
async def retrieve_from_chroma_tool(query: str) -> List[str]:
    """Retrieves relevant documents from the local ChromaDB vector store."""
    print(f"--- Tool Call: retrieve_from_chroma for query: '{query}' ---")
    query_embedding = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return results['documents'][0]

async def grade_documents_tool(question: str, documents: List[str]) -> List[str]:
    """Grades documents for relevance and returns only the relevant ones."""
    print("--- Tool Call: grade_documents ---")
    tasks = [grade_relevance_async(question, doc) for doc in documents]
    results = await asyncio.gather(*tasks)
    
    filtered_docs = [doc for doc, score in zip(documents, results) if score.get("score", "no").lower() == "yes"]
    print(f"--- GRADE: {len(filtered_docs)}/{len(documents)} DOCUMENTS RELEVANT ---")
    return filtered_docs

async def web_search_tool(question: str) -> str:
    """Performs a web search using Tavily."""
    print("--- Tool Call: web_search ---")
    response = await asyncio.to_thread(tavily_client.search, query=question, search_depth="basic")
    return "\n".join([d["content"] for d in response.get("results", [])])

async def generate_answer_tool(question: str, context: List[str]) -> str:
    """Generates a final answer based on the provided context."""
    print("--- Tool Call: generate_answer ---")
    return await generate_answer_from_context_async(question, context)

async def grade_hallucination_tool(answer: str, documents: List[str]) -> str:
    """Grades if the answer is supported by the documents."""
    print("--- Tool Call: grade_hallucination ---")
    result = await grade_hallucination_async(answer, documents)
    score = result.get("score", "no").lower()
    print(f"--- HALLUCINATION GRADE: {score.upper()} ---")
    return score

# --- Instantiating ADK FunctionTools ---
tools = [
    FunctionTool(func=retrieve_from_chroma_tool),
    FunctionTool(func=grade_documents_tool),
    FunctionTool(func=web_search_tool),
    FunctionTool(func=generate_answer_tool),
    FunctionTool(func=grade_hallucination_tool)
]

# =============================================================================
# 5. DEFINING THE SELF-RAG AGENT
# =============================================================================
self_rag_agent = LlmAgent(
    name="Self_RAG_Agent",
    model=AGENT_LLM,
    description="A self-correcting agent that answers questions using internal and external knowledge.",
    tools=tools,
    instruction="""
    You are an expert research assistant. Your goal is to answer the user's question with verified information. Follow this workflow precisely:

    **Step 1: Retrieve and Grade Documents**
    1. Call `retrieve_from_chroma_tool` with the user's `question` to get initial documents.
    2. Call `grade_documents_tool` with the `question` and the retrieved `documents` to filter out irrelevant ones.

    **Step 2: Initial Answer Generation**
    3. Call `generate_answer_tool` using the `question` and the filtered (graded) `documents`.

    **Step 3: Self-Correction via Hallucination Check**
    4. Call `grade_hallucination_tool` with the `answer` from Step 3 and the same filtered documents.
    5. **Analyze the result:**
        *   If "yes", the answer is supported. This is your final answer. STOP and provide the answer.
        *   If "no", the answer is unsupported. The local knowledge is insufficient. Proceed to Step 4.

    **Step 4: Fallback to Web Search (if needed)**
    6. Call `web_search_tool` with the original `question`.
    7. Combine the web search results with the original filtered documents to create a new, richer context.
    8. Call `generate_answer_tool` one final time with the `question` and this new combined context. This is your definitive answer.
    """,
)

# =============================================================================
# 6. MAIN EXECUTION FUNCTION
# =============================================================================
async def run_self_rag(question: str):
    """Runs the Self-RAG agent workflow."""
    session_service = InMemorySessionService()
    print(f"\n🚀 Starting Self-RAG Workflow for question: '{question}'...")

    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=self_rag_agent, app_name=APP_NAME, session_service=session_service)

    initial_message = types.Content(role='user', parts=[types.Part(text=question)])
    final_answer = None

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=initial_message):
            print(f"--- Agent State: {event.author} is thinking... ---")
            if event.is_final_response():
                final_answer = event.content.parts[0].text
                break
    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        return None

    print("\n🎉 Self-RAG Workflow Finished!")
    return final_answer

# =============================================================================
# 7. SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Ensures that file paths are relative to the script's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    async def main():
        # Example 1: Relevant question (should be found in the document)
        relevant_question = "What was the Logic Theorist?"
        result_1 = await run_self_rag(relevant_question)
        if result_1:
            print("\n✅ Final Answer (Relevant Question):")
            print("---------------------------------")
            print(result_1)
        
        print("\n" + "="*80 + "\n")

        # Example 2: Irrelevant question (should trigger a web search)
        irrelevant_question = "Who is the current CEO of Microsoft?"
        result_2 = await run_self_rag(irrelevant_question)
        if result_2:
            print("\n✅ Final Answer (Irrelevant Question):")
            print("---------------------------------")
            print(result_2)
    
    asyncio.run(main())