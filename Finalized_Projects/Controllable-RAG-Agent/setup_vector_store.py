# setup_generic_pdf.py

import os
import PyPDF2
from dotenv import load_dotenv

# --- Local Imports ---
from config import OLLAMA_EMBEDDING_MODEL

# --- LangChain Community Imports ---
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# --- Setup ---
load_dotenv()

# --- !!! IMPORTANT !!! ---
# --- Change this path to point to your PDF file ---
PDF_PATH = "data/20250312_013613_Job-Skills-Report-2025.pdf"
VECTOR_STORE_DIR = "pdf_chunks_vectorstore"

def main():
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF file not found at '{PDF_PATH}'")
        print("Please update the PDF_PATH variable in this script.")
        return

    print(f"\n--- 1. Loading and Processing PDF: '{PDF_PATH}' ---")
    
    # Load the PDF and extract text from each page
    try:
        loader = PyPDF2.PdfReader(PDF_PATH)
        documents = []
        for i, page in enumerate(loader.pages):
            page_content = page.extract_text()
            if page_content: # Ensure page is not empty
                doc = Document(page_content=page_content, metadata={"page": i + 1})
                documents.append(doc)
        print(f"📄 Loaded {len(documents)} pages from the PDF.")
    except Exception as e:
        print(f"❌ Failed to read or process the PDF file: {e}")
        return

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"✂️ Split the document into {len(texts)} text chunks.")
    
    # --- 2. Create and Save the FAISS Vector Store using Ollama ---
    print(f"\n--- 2. Creating Vector Store with Ollama ('{OLLAMA_EMBEDDING_MODEL}') ---")
    try:
        # Use OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(VECTOR_STORE_DIR)
        print(f"✅ Vector store created and saved successfully to '{VECTOR_STORE_DIR}'.")
    except Exception as e:
        print(f"❌ An error occurred during vector store creation: {e}")
        print("   Please ensure the Ollama application is running and the model is pulled.")
        return
        
    print("\n🎉 Generic PDF setup complete!")

if __name__ == "__main__":
    main()