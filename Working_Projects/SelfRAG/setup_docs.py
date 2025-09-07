# setup_docs.py

import os
import PyPDF2
from dotenv import load_dotenv
import chromadb
import ollama

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# --- Configuration ---
load_dotenv()
INPUTS_DIR = "inputs"
CHROMA_DB_PATH = "local_chroma_db"
COLLECTION_NAME = "local_knowledge"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

def split_text_into_chunks(text: str) -> list[str]:
    """A simple custom splitter that splits a document by paragraphs."""
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    return chunks

def get_all_documents_from_inputs() -> list[str]:
    """
    Reads all .txt and .pdf files from the INPUTS_DIR, extracts their text,
    and returns a list of text content.
    """
    all_texts = []
    print(f"📂 Reading documents from '{INPUTS_DIR}' directory...")
    
    for filename in os.listdir(INPUTS_DIR):
        file_path = os.path.join(INPUTS_DIR, filename)
        
        if filename.endswith('.pdf'):
            try:
                print(f"  📄 Processing PDF: {filename}")
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += page_text + "\n"
                    all_texts.append(pdf_text)
            except Exception as e:
                print(f"    - ❌ Error reading PDF {filename}: {e}")

        elif filename.endswith('.txt'):
            try:
                print(f"  📄 Processing TXT: {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_texts.append(f.read())
            except Exception as e:
                print(f"    - ❌ Error reading TXT {filename}: {e}")
        else:
            print(f"  ⚠️ Skipping unsupported file type: {filename}")
            
    return all_texts

def main():
    if not os.path.exists(INPUTS_DIR):
        print(f"❌ Error: Directory not found at '{INPUTS_DIR}'")
        print("Please create the 'inputs' directory and add your .txt or .pdf files.")
        return

    # --- 1. Load and Process all Documents ---
    document_contents = get_all_documents_from_inputs()
    if not document_contents:
        print("❌ No documents found or processed. Aborting.")
        return
        
    print("\n--- 2. Splitting all documents into chunks ---")
    all_chunks = []
    for text in document_contents:
        all_chunks.extend(split_text_into_chunks(text))
    print(f"✂️ Total chunks created: {len(all_chunks)}")

    # --- 3. Create and Save the ChromaDB Vector Store ---
    print(f"\n--- 3. Creating ChromaDB Store with Ollama ('{OLLAMA_EMBEDDING_MODEL}') ---")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Delete the collection if it exists to ensure a fresh start
        if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
            print(f"   - Deleting existing collection '{COLLECTION_NAME}'...")
            chroma_client.delete_collection(name=COLLECTION_NAME)
            
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        
        print("   - Generating embeddings for each chunk (this may take a moment)...")
        embeddings = []
        doc_ids = [f"doc_chunk_{i}" for i in range(len(all_chunks))]
        
        for i, chunk in enumerate(all_chunks):
            response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=chunk)
            embeddings.append(response["embedding"])
            if (i + 1) % 50 == 0:
                print(f"     ...embedded {i + 1}/{len(all_chunks)} chunks...")

        collection.add(embeddings=embeddings, documents=all_chunks, ids=doc_ids)
        print(f"✅ Vector store created and saved successfully to '{CHROMA_DB_PATH}'.")
        
    except Exception as e:
        print(f"❌ An error occurred during vector store creation: {e}")
        print("   Please ensure the Ollama application is running and the model is pulled.")
        return
        
    print("\n🎉 Document ingestion complete!")

if __name__ == "__main__":
    main()