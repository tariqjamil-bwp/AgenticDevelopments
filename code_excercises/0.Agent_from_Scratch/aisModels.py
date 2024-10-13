import os
from groq import Groq

class GroqLLM:
    def __init__(self, api_key=None, model="llama3-70b-8192"):
        """
        Initialize the LLM with a Groq client and a model name.

        :param api_key: The API key for the Groq client.
        :param model: The model to use for generating responses. Default is 'llama3-70b-8192'.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or through the 'GROQ_API_KEY' environment variable.")
        
        # Initialize the Groq client with the provided API key
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate_response(self, messages):
        """
        Generate a response from the LLM based on the provided conversation history.

        :param messages: A list of messages representing the conversation history.
        :return: The generated response text.
        """
        if isinstance(messages, dict):
            # If a single message is passed as a dict, wrap it in a list
            messages = [messages]

        if isinstance(messages, str):
            # If the input is a single string, convert it into a list of messages
            messages = [{"role": "user", "content": messages}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            # Extract and return the assistant's reply
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
# Usage example
if __name__ == "__main__":
    import os
    llm = GroqLLM()
    os.system('clear')
    # Generating a response with a single user message
    #response = llm.generate_response({'role': 'user', 'content': 'Explain the importance of fast language models'})
    response = llm.generate_response('Explain the importance of fast language models')
    
    
    print(response)
