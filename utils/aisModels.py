import os
from groq import Groq
from langchain_core.runnables import Runnable

class GroqLLM(Runnable):
    def __init__(self, api_key=None, model="llama3-70b-8192"):
        """
        Initialize the LLM with a Groq client and a model name.
        
        :param api_key: The API key for the Groq client.
        :param model: The model to use for generating responses. Default is 'llama3-70b-8192'.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
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
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return completion.choices[0].message.content

    def invoke(self, input):
        """
        A wrapper method for generating a response using the provided input.
        
        :param input: A single string or a list of strings as input.
        :return: The generated response text(s) as a string or a list of strings.
        """
        # Handle case where input is a string
        if isinstance(input, str):
            return self.generate_response(input)
        
        if isinstance(input, dict):
            input = input.get("query")
            result = self.generate_response(input)
            return {"result": result}  # Return as a dictionary

        # Handle case where input is a list of strings
        elif isinstance(input, list):
            responses = []
            for message in input:
                response = self.generate_response(message)
                responses.append(response)
            return '\n'.join(responses) if len(responses) > 1 else ''.join(responses)

        elif isinstance(input, list):
            if all(isinstance(message, dict) for message in input):
                responses = {}
                for message in input:
                    # Assuming `generate_response` is a method that processes the dictionary
                    response = self.generate_response(message)
                    # You can use the dictionary keys as response keys, or modify as needed
                    responses.update(response)
                return responses
        else:
            raise ValueError("Each item in the input list must be a dictionary.")

# Usage example
if __name__ == "__main__":
    os.system('clear')
    
    # Initialize the GroqLLM instance
    llm = GroqLLM()
    
    # Generate a response for a single input string
    response = llm.invoke("Explain the importance of fast language models.")
    print("Single response:", response)
    
    # Generate responses for a list of input strings
    input_list = [
        "Explain the importance of fast language models.",
        "What is the role of transformers in NLP?",
        "How can Groq optimize model performance?"
    ]
    responses = llm.invoke(input_list)
    print("List of responses:", responses)
