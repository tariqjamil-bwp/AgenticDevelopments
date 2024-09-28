import os
from langchain_groq import ChatGroq

class GroqLLM(ChatGroq):
    def __init__(self, temperature=0, model_name="llama3-70b-8192", api_key=None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY must be provided either as an argument or in the environment variables.")
        
        # Directly call the superclass (ChatGroq) initialization
        super().__init__(
            temperature=temperature,
            model_name=model_name,
            api_key=api_key
        )


os.system("clear")

llm = GroqLLM()
print(llm.invoke("hello").content)

from langchain_ollama import ChatOllama

class OllamaLLM(ChatOllama):
    def __init__(self, temperature=0, model_name="llama3.1"):
        # Directly call the superclass (ChatOllama) initialization with the fixed base URL
        super().__init__(
            temperature=temperature,
            model=model_name,
        )
#pip install langchain-huggingface 
embedder={
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }
 }

# Example usage
llm = OllamaLLM()
print(llm.invoke("hello").content)


