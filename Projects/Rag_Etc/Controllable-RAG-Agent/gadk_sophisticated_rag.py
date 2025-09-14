# adk_generic_rag.py

import os
import asyncio

# --- Local Imports ---
from config import OLLAMA_EMBEDDING_MODEL, GEMINI_LLM_MODEL

# --- ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.genai import types

# --- LangChain & Helper Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# =============================================================================
# SETUP: ENVIRONMENT AND CONSTANTS
# =============================================================================
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

APP_NAME = "generic_rag_app_gemini"
USER_ID = "rag_user_004"
SESSION_ID = "rag_session_004"
VECTOR_STORE_DIR = "pdf_chunks_vectorstore"

# =============================================================================
# INITIALIZING MODELS, VECTOR STORE, AND RETRIEVER
# =============================================================================
AGENT_MODEL = LiteLlm(GEMINI_LLM_MODEL)
ANSWERING_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

print("Loading FAISS vector store created with Ollama...")
try:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print(f"✅ Vector store '{VECTOR_STORE_DIR}' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    print(f"Please run the `setup_generic_pdf.py` script first.")
    exit()

# =============================================================================
# DEFINING LANGCHAIN CHAIN FOR THE "ANSWER" TOOL
# =============================================================================
class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")

ANSWER_COT_TEMPLATE = """
Answer the following question based *only* on the provided context.
Provide a clear, concise answer. If the context is insufficient to answer the question, state that the information is not available in the provided document.

Context:
---
{context}
---

Question: {question}
"""
ANSWER_COT_PROMPT = PromptTemplate(
    template=ANSWER_COT_TEMPLATE,
    input_variables=["context", "question"],
)
ANSWERING_CHAIN = ANSWER_COT_PROMPT | ANSWERING_LLM.with_structured_output(QuestionAnswerFromContext)

# =============================================================================
# DEFINING ADK TOOLS
# =============================================================================

async def retrieve_pdf_chunks_tool(query: str) -> str:
    """Retrieves relevant text chunks from the loaded PDF document based on a query."""
    print(f"Tool Call: Retrieving PDF chunks for query: '{query}'")
    docs = await asyncio.to_thread(retriever.get_relevant_documents, query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

async def answer_from_context_tool(question: str, context: str) -> str:
    """Answers a question using only the provided context."""
    print(f"Tool Call: Synthesizing answer for question: '{question}'")
    input_data = {"question": question, "context": context}
    output = await asyncio.to_thread(ANSWERING_CHAIN.invoke, input_data)
    return output.answer_based_on_content

tool_retrieve_chunks = FunctionTool(func=retrieve_pdf_chunks_tool)
tool_answer_question = FunctionTool(func=answer_from_context_tool)

# =============================================================================
# DEFINING THE RAG AGENT
# =============================================================================
generic_rag_agent = LlmAgent(
    name="Generic_RAG_Agent",
    model=AGENT_MODEL,
    description="Answers questions about a document using a retrieve-then-answer workflow.",
    tools=[
        tool_retrieve_chunks,
        tool_answer_question,
    ],
    instruction="""
    You are an expert document analysis agent. Your goal is to answer the user's question accurately.

    ### Your Workflow:
    1.  **Retrieve:** Use the `retrieve_pdf_chunks_tool` with a query that best represents the user's question to find relevant context from the document.
    2.  **Answer:** Use the `answer_from_context_tool`. Pass the original user question and the context you retrieved in the previous step to this tool.
    3.  **Final Response:** The output of the `answer_from_context_tool` is your final answer. Do not add any extra text or commentary.
    """,
)

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
async def run_generic_rag(question: str):
    """Runs the generic RAG agent workflow."""
    session_service = InMemorySessionService()
    print(f"🚀 Starting Generic RAG Workflow for question: '{question}'...")

    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=generic_rag_agent, app_name=APP_NAME, session_service=session_service)

    initial_message = types.Content(role='user', parts=[types.Part(text=question)])
    final_answer = None

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=initial_message):
            if event.is_final_response():
                final_answer = event.content.parts[0].text
                break
    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        return None

    print("\n🎉 Generic RAG Workflow Finished!")
    return final_answer

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # --- !!! IMPORTANT !!! ---
    # --- Change this question to be relevant to your PDF document ---
    user_question = "What are the top 3 skills in demand for 2025 according to the report?. Give bullited list with one liner details."

    final_result = asyncio.run(run_generic_rag(user_question))

    if final_result:
        print("\n✅ Final Answer:")
        print("---------------------------------")
        print(final_result)
        print("---------------------------------")
    else:
        print("\n⚠️ Workflow did not produce a final answer.")