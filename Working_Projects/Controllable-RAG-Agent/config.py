# config.py
# Storing model and embedding configurations for the RAG agent.

# --- Embedding Model Configuration ---
# Specifies the model Ollama will use to create embeddings.
# Make sure Ollama is running and has this model pulled.
# Example command: ollama pull nomic-embed-text
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


# --- Language Model Configuration ---
# Specifies the Gemini model for the agent's reasoning and answering.
# This requires a GOOGLE_API_KEY to be set in your environment.
GEMINI_LLM_MODEL = "gemini/gemini-1.5-flash"