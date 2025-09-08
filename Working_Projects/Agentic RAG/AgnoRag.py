# ==============================================================================
# Imports
# ==============================================================================
import logging
import os
import tempfile
from datetime import datetime
from typing import List
import streamlit as st
import bs4
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.exa import ExaTools
from agno.embedder.ollama import OllamaEmbedder as AgnoOllamaEmbedder
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- FAISS CHANGE: Replace Qdrant with FAISS ---
from langchain_community.vectorstores import FAISS
# -----------------------------------------------
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ==============================================================================
# Logging Configuration
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ==============================================================================
# Environment and Session State Configuration
# ==============================================================================
if 'app_initialized' not in st.session_state:
    logger.info("--- New Session: Initializing App State ---")
    load_dotenv()
    st.session_state.exa_api_key = os.getenv("EXA_API_KEY", "")
    st.session_state.model_version = "deepseek-r1:1.5b"
    st.session_state.vector_store = None
    st.session_state.processed_documents = []
    st.session_state.history = []
    st.session_state.use_web_search = False
    st.session_state.force_web_search = False
    st.session_state.similarity_threshold = 0.7
    st.session_state.rag_enabled = True
    st.session_state.app_initialized = True
else:
    logger.info("--- Page Reload: App State Already Initialized ---")

# ==============================================================================
# Embedding Class (Remains the same)
# ==============================================================================
class OllamaEmbedder(Embeddings):
    def __init__(self, model_name="snowflake-arctic-embed"):
        self.embedder = AgnoOllamaEmbedder(id=model_name, dimensions=1024)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text: str) -> List[float]:
        return self.embedder.get_embedding(text)

# ==============================================================================
# Streamlit App UI
# ==============================================================================
st.title("🐋 Deepseek Local RAG Reasoning Agent")

# --- Sidebar Configuration ---
st.sidebar.header("🤖 Agent Configuration")
st.session_state.model_version = st.sidebar.radio("Select Model Version", ["deepseek-r1:1.5b", "deepseek-r1:7b"], index=0 if st.session_state.model_version == "deepseek-r1:1.5b" else 1)
st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG Mode", value=st.session_state.rag_enabled)

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.session_state.vector_store = None # Clear vector store on reset
    st.session_state.processed_documents = []
    st.rerun()

# --- FAISS CHANGE: Removed all Qdrant UI elements ---
if st.session_state.rag_enabled:
    st.sidebar.header("🎯 Search Configuration")
    st.session_state.similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, st.session_state.similarity_threshold)

st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search", value=st.session_state.use_web_search)
if st.session_state.use_web_search:
    st.session_state.exa_api_key = st.sidebar.text_input("Exa AI API Key", type="password", value=st.session_state.exa_api_key)
    search_domains_str = st.sidebar.text_input("Searchable domains", value="arxiv.org,wikipedia.org,github.com,medium.com")
    search_domains = [d.strip() for d in search_domains_str.split(",") if d.strip()]

# ==============================================================================
# Utility Functions
# ==============================================================================
def process_documents(file, source_type: str) -> List:
    logger.info(f"Processing document. Source type: {source_type}")
    try:
        if source_type == 'pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                loader = PyPDFLoader(tmp_file.name)
            metadata = {"source_type": "pdf", "file_name": file.name}
        elif source_type == 'web':
            loader = WebBaseLoader(web_paths=(file,), bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header", "content", "main"))))
            metadata = {"source_type": "url", "url": file}
        else: return []
        
        documents = loader.load()
        for doc in documents: doc.metadata.update({**metadata, "timestamp": datetime.now().isoformat()})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        logger.info(f"Successfully processed and split document into {len(splits)} chunks.")
        return splits
    except Exception as e:
        logger.error(f"DOCUMENT PROCESSING FAILED: {e}", exc_info=True)
        st.error(f"📄 Document processing error: {e}")
        return []

# ==============================================================================
# Vector Store Management (REVISED FOR FAISS)
# ==============================================================================
def manage_vector_store(texts: List):
    logger.info("Managing FAISS vector store in session state.")
    try:
        with st.spinner('Embedding documents... This may take a moment on your local machine.'):
            # Get the embedding model
            embeddings = OllamaEmbedder()
            
            # If a vector store already exists, add new documents to it
            if st.session_state.vector_store is not None:
                logger.info(f"Adding {len(texts)} new documents to existing FAISS store.")
                st.session_state.vector_store.add_documents(texts, embedding=embeddings)
            # If no vector store exists, create a new one from the documents
            else:
                logger.info(f"Creating new FAISS store from {len(texts)} documents.")
                st.session_state.vector_store = FAISS.from_documents(texts, embedding=embeddings)
        
        st.success("✅ Documents embedded and stored successfully!")
        logger.info("FAISS vector store updated successfully.")
    except Exception as e:
        logger.error(f"FAISS VECTOR STORE OPERATION FAILED: {e}", exc_info=True)
        st.error(f"🔴 Vector store error: {e}")

# ==============================================================================
# Agent Initialization (Remains mostly the same)
# ==============================================================================
def get_agent(agent_type="rag"):
    global search_domains
    tools_list = []
    if agent_type == "web" and 'search_domains' in locals():
        tools_list = [ExaTools(api_key=st.session_state.exa_api_key, include_domains=search_domains)]
        
    return Agent(
        model=Ollama(id=st.session_state.model_version),
        tools=tools_list,
        instructions="You are an intelligent AI assistant. Answer questions based on provided context.",
        show_tool_calls=True, markdown=True
    )

# ==============================================================================
# Main Application Logic (UPDATED FOR FAISS)
# ==============================================================================
if st.session_state.rag_enabled:
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")
    
    if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
        with st.spinner(f'Processing {uploaded_file.name}...'):
            texts = process_documents(uploaded_file, 'pdf')
            if texts:
                manage_vector_store(texts)
                st.session_state.processed_documents.append(uploaded_file.name)

    if web_url and web_url not in st.session_state.processed_documents:
        with st.spinner(f'Processing {web_url}...'):
            texts = process_documents(web_url, 'web')
            if texts:
                manage_vector_store(texts)
                st.session_state.processed_documents.append(web_url)

# --- Chat Interface ---
chat_col, toggle_col = st.columns([0.9, 0.1])
with chat_col: prompt = st.chat_input("Ask about your documents..." if st.session_state.rag_enabled else "Ask me anything...")
with toggle_col: st.session_state.force_web_search = st.toggle('🌐', help="Force web search for this query", value=st.session_state.force_web_search)

# Display chat history
for message in st.session_state.history:
    with st.chat_message(message["role"]): st.markdown(message["content"])

# --- Main chat logic ---
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    context, docs = "", []
    if st.session_state.rag_enabled:
        use_web_search = st.session_state.force_web_search
        if not use_web_search and st.session_state.vector_store:
            with st.spinner("🔍 Searching documents locally with FAISS..."):
                retriever = st.session_state.vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": st.session_state.similarity_threshold})
                docs = retriever.invoke(prompt)
                if docs:
                    context = "\n\n".join([d.page_content for d in docs])
                    st.info(f"📊 Found {len(docs)} relevant document chunks.")
                else:
                    st.info("🔄 No relevant documents found in the local vector store.")
                    use_web_search = True
        elif not use_web_search:
             st.warning("RAG is enabled, but no documents have been uploaded yet.")
             use_web_search = True

        if use_web_search and st.session_state.use_web_search and st.session_state.exa_api_key:
            with st.spinner("🌐 Searching the web..."):
                try:
                    web_agent = get_agent("web")
                    web_results = web_agent.run(prompt).content
                    context = f"Web Search Results:\n{web_results}"
                except Exception as e:
                    logger.error(f"WEB SEARCH FAILED: {e}", exc_info=True)
                    st.error(f"❌ Web search failed: {e}")

    with st.spinner("🤖 Thinking..."):
        try:
            rag_agent = get_agent("rag")
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
            response_content = rag_agent.run(full_prompt).content.strip()

            st.session_state.history.append({"role": "assistant", "content": response_content})
            with st.chat_message("assistant"):
                st.markdown(response_content)
                if docs and not st.session_state.force_web_search:
                    with st.expander("🔍 See document sources"):
                        for doc in docs:
                            source_info = doc.metadata.get("file_name", doc.metadata.get("url", "Unknown"))
                            st.write(f"**Source:** `{source_info}`")
                            st.write(f"> {doc.page_content[:250].replace('\n', ' ')}...")
                            
        except Exception as e:
            logger.error(f"RESPONSE GENERATION FAILED: {e}", exc_info=True)
            st.error(f"❌ Error during response generation: {e}")

elif not st.session_state.history:
    if st.session_state.rag_enabled: st.info("Welcome! Upload a PDF or enter a URL to begin chatting with your documents.")
    else: st.info("Welcome! RAG mode is disabled. You can ask me anything.")