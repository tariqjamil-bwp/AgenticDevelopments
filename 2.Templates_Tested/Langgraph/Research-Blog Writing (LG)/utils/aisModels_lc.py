import os
from langchain_groq import ChatGroq
class GroqChatLLM(ChatGroq):
    def __init__(self, temperature=0, model_name="llama-3.1-70b-versatile", api_key=None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY must be provided either as an argument or in the environment variables.")
        
        # Directly call the superclass (ChatGroq) initialization
        super().__init__(
            temperature=temperature,
            model_name=model_name,
            api_key=api_key
        )


from langchain_ollama import ChatOllama
class OllamaLLM(ChatOllama):
    def __init__(self, temperature=0, model_name="llama3.1"):
        # Directly call the superclass (ChatOllama) initialization with the fixed base URL
        super().__init__(
            temperature=temperature,
            model=model_name,
        )

if __name__ == "__main__":
        
    # Example usage
    os.system("clear")

    llm = GroqChatLLM()
    print(llm.invoke("hello").content)

    llm = OllamaLLM()
    print(llm.invoke("hello").content)


