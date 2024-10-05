import os
from dotenv import load_dotenv
x = load_dotenv()
from groq import Groq
class GroqLLM:
    def __init__(self, api_key="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", model="llama3-70b-8192"):
        """
        Initialize the LLM with a Groq client and a model name.
        
        :param api_key: The API key for the Groq client.
        :param model: The model to use for generating responses. Default is 'llama3-70b-8192'.
        """
        self.api_key = api_key #or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or through the 'GROQ_API_KEY' environment variable.")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate_response(self, messages):
        """
        Generate a response from the LLM based on the provided conversation history.
        
        :param messages: A list of messages representing the conversation history.
        :return: The generated response text.
        """
        if isinstance(messages, str):
            # If the input is a single string, convert it into a list of messages
            messages = [{"role": "user", "content": messages}]
        
        if isinstance(messages, dict):
            messages = [messages]
        
        if not isinstance(messages, list):
            raise ValueError("Input must be a string, dict, or list of messages.")
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return completion.choices[0].message.content

# Usage example
if __name__ == "__main__":
    import os
    os.system('clear')
    llm = GroqLLM()
    response = llm.generate_response("Explain the importance of fast language models")
    print(response)
