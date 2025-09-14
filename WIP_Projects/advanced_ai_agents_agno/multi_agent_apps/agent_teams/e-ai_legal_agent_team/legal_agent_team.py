# -*- coding: utf-8 -*-
"""
Legal Document Analyzer using Google ADK with Streamlit
Uses constant credentials, integrates PDFKnowledgeBase, Qdrant, and DuckDuckGo
"""

import streamlit as st
import os
import asyncio
import tempfile
from textwrap import dedent
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.document.chunking.document import DocumentChunking
from agno.embedder.ollama import OllamaEmbedder
from ddgs import DDGS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
from datetime import datetime

# Suppress OpenTelemetry warnings
import logging
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants for credentials
LITELLM_API_KEY = os.getenv("GEMINI_API_KEY") or "your-litellm-api-key"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or "your-qdrant-api-key"
QDRANT_URL = os.getenv("QDRANT_URL") or "your-qdrant-url"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "nomic-embed-text"
COLLECTION_NAME = "legal_documents"
APP_NAME = "legal_document_analyzer"
USER_ID = "legal_user"
SESSION_ID = "legal_session"
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

# Set environment variable for LiteLLM
os.environ["LITELLM_API_KEY"] = LITELLM_API_KEY

# LLM Model
AGENT_MODEL = LiteLlm("gemini/gemini-2.0-flash")

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

# DuckDuckGo search tool
def ddg_search(query: str) -> str:
    """Wrapper to search using DuckDuckGo"""
    if not query or query.strip() == "":
        return "ERROR: No valid search query provided."
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
        if not results:
            return "No results found."
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['href']} - {result['title']}")
        return "\n".join(output)
    except Exception as e:
        return f"Search failed: {str(e)}"

ddg_tool = FunctionTool(
    func=ddg_search
)

# Initialize Qdrant
def init_qdrant():
    """Initialize Qdrant client with configured settings."""
    try:
        vector_db = Qdrant(
            collection=COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            embedder=OllamaEmbedder(
                id=OLLAMA_MODEL
            )
        )
        return vector_db
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None

def process_document(uploaded_file, vector_db: Qdrant):
    """Process document, create embeddings and store in Qdrant vector database"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        st.info("Loading and processing document...")
        
        knowledge_base = PDFKnowledgeBase(
            temp_file_path,
            vector_db=vector_db,
            reader=PDFReader(),
            chunking_strategy=DocumentChunking(
                chunk_size=1000,
                overlap=200
            )
        )
        if knowledge_base:
            st.success("✅ Documents processed successfully!")
        else:
            st.error("Error processing documents")
        
        with st.spinner('📤 Loading documents into knowledge base...'):
            try:
                knowledge_base.load(recreate=True, upsert=True)
                st.success("✅ Documents stored successfully!")
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
                raise
        
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass
            
        return knowledge_base
            
    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")

async def run_agent(agent, query):
    """Helper function to run an ADK agent and return the response."""
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response():
            return event.content.parts[0].text
    return "No response generated."

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 Configuration")
        if not all([LITELLM_API_KEY, QDRANT_API_KEY, QDRANT_URL, OLLAMA_MODEL]):
            st.error("Missing required credentials. Check environment variables.")
            return
        st.success("Configuration loaded!")

        if not st.session_state.vector_db:
            st.session_state.vector_db = init_qdrant()
            if st.session_state.vector_db:
                st.success("Successfully connected to Qdrant!")

        st.divider()
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf'])
        
        if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
            with st.spinner("Processing document..."):
                try:
                    knowledge_base = process_document(uploaded_file, st.session_state.vector_db)
                    
                    if knowledge_base:
                        st.session_state.knowledge_base = knowledge_base
                        st.session_state.processed_files.add(uploaded_file.name)
                        
                        legal_researcher = LlmAgent(
                            name="LegalResearcher",
                            model=AGENT_MODEL,
                            description="Legal research specialist",
                            instruction=dedent(
                                f"""\
                                INSTRUCTIONS:
                                - Find and cite relevant legal cases and precedents using `ddg_search`.
                                - Provide detailed research summaries with sources.
                                - Reference specific sections from the uploaded document in the knowledge base.
                                - Always search the knowledge base for relevant information.
                                - Output in markdown format.
                                - Date: {CURRENT_DATE}
                                """
                            ),
                            tools=[ddg_tool],
                            output_key="research_summary"
                        )

                        contract_analyst = LlmAgent(
                            name="ContractAnalyst",
                            model=AGENT_MODEL,
                            description="Contract analysis specialist",
                            instruction=dedent(
                                f"""\
                                INSTRUCTIONS:
                                - Review contracts thoroughly.
                                - Identify key terms and potential issues.
                                - Reference specific clauses from the document in the knowledge base.
                                - Output in markdown format.
                                - Date: {CURRENT_DATE}
                                """
                            ),
                            output_key="contract_analysis"
                        )

                        legal_strategist = LlmAgent(
                            name="LegalStrategist",
                            model=AGENT_MODEL,
                            description="Legal strategy specialist",
                            instruction=dedent(
                                f"""\
                                INSTRUCTIONS:
                                - Develop comprehensive legal strategies.
                                - Provide actionable recommendations.
                                - Consider both risks and opportunities.
                                - Reference the knowledge base for document-specific insights.
                                - Output in markdown format.
                                - Date: {CURRENT_DATE}
                                """
                            ),
                            output_key="strategy_recommendations"
                        )

                        st.session_state.legal_team = SequentialAgent(
                            name="LegalTeamLead",
                            description="Coordinates legal analysis",
                            sub_agents=[legal_researcher, contract_analyst, legal_strategist],
                            instruction=dedent(
                                f"""\
                                INSTRUCTIONS:
                                - Coordinate analysis between Legal Researcher, Contract Analyst, and Legal Strategist.
                                - Provide comprehensive responses combining insights from all agents.
                                - Ensure all recommendations are properly sourced and reference the knowledge base.
                                - Output in markdown format with sections: Detailed Analysis, Key Points, Recommendations.
                                - Date: {CURRENT_DATE}
                                """
                            )
                        )
                        
                        st.success("✅ Document processed and team initialized!")
                        
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        elif uploaded_file:
            st.success("✅ Document already processed and team ready!")

        st.divider()
        st.header("🔍 Analysis Options")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Contract Review",
                "Legal Research",
                "Risk Assessment",
                "Compliance Check",
                "Custom Query"
            ]
        )

    if not all([LITELLM_API_KEY, st.session_state.vector_db]):
        st.info("Configuration or Qdrant connection missing. Check environment variables.")
    elif not uploaded_file:
        st.info("👈 Please upload a legal document to begin analysis")
    elif st.session_state.legal_team:
        analysis_icons = {
            "Contract Review": "📑",
            "Legal Research": "🔍",
            "Risk Assessment": "⚠️",
            "Compliance Check": "✅",
            "Custom Query": "💭"
        }

        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
  
        analysis_configs = {
            "Contract Review": {
                "query": "Review this contract and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
                "description": "Detailed contract analysis focusing on terms and obligations"
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to this document.",
                "agents": ["Legal Researcher"],
                "description": "Research on relevant legal cases and precedents"
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities in this document.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Compliance Check": {
                "query": "Check this document for regulatory compliance issues.",
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Comprehensive compliance analysis"
            },
            "Custom Query": {
                "query": None,
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Custom analysis using all available agents"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 Active Legal AI Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")

        if analysis_type == "Custom Query":
            user_query = st.text_area(
                "Enter your specific query:",
                help="Add any specific questions or points you want to analyze"
            )
        else:
            user_query = None

        if st.button("Analyze"):
            if analysis_type == "Custom Query" and not user_query:
                st.warning("Please enter a query")
            else:
                with st.spinner("Analyzing document..."):
                    try:
                        if analysis_type != "Custom Query":
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            Please search the knowledge base and provide specific references from the document.
                            """
                        else:
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            {user_query}
                            
                            Please search the knowledge base and provide specific references from the document.
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        # Run the legal team agent
                        response = asyncio.run(run_agent(st.session_state.legal_team, combined_query))
                        
                        tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        
                        with tabs[0]:
                            st.markdown("### Detailed Analysis")
                            st.markdown(response)
                        
                        with tabs[1]:
                            st.markdown("### Key Points")
                            key_points_query = f"""
                            Based on this previous analysis:
                            {response}
                            
                            Please summarize the key points in bullet points.
                            Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """
                            key_points_response = asyncio.run(run_agent(st.session_state.legal_team, key_points_query))
                            st.markdown(key_points_response)
                        
                        with tabs[2]:
                            st.markdown("### Recommendations")
                            recommendations_query = f"""
                            Based on this previous analysis:
                            {response}
                            
                            What are your key recommendations based on the analysis, the best course of action?
                            Provide specific recommendations from: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """
                            recommendations_response = asyncio.run(run_agent(st.session_state.legal_team, recommendations_query))
                            st.markdown(recommendations_response)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()