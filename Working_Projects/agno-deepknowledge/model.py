from phi.model.groq import Groq
from phi.model.openai import OpenAI
import os
groq_api_key = os.getenv('GROQ_API_KEY')

def get_client(model_type, model):
    if model_type=='groq':        
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key= groq_api_key)
        model = 'deepseek-r1-distill-llama-70b' 
        #model = model
    elif model_type=='ollama':
        client = OpenAI(base_url = 'http://localhost:11434/v1', api_key='ollama',) # required, but unused
        model = 'deepseek-r1'
        
    elif model_type=='mistral':
        client = OpenAI(base_url = "https://api.braintrust.dev/v1/proxy", api_key=mistral_api_key)
        model = 'mistral-large-latest'
    
    elif model_type == "phi-groq":
        model=model if model else 'deepseek-r1-distill-llama-70b' 
        client = Groq(api_key=groq_api_key, id=model)
        
    elif model_type =='phi-groq-2':
        client = OpenAIChat(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)
        model = 'deepseek-r1-distill-llama-70b' 
    
    elif model_type =='phi-gemini':
        client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=GEMINI_API_KEY)
        model = 'gemini-2.0-flash' 
    
    return client, model

def get_client_openrouter(model=None):
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
        )
    
    model = "qwen/qwen2.5-vl-72b-instruct:free"
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What's in this image?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
                }
            ]
            }
        ]
        )
    print(completion.choices[0].message.content)
    
    
def get_client_gemini():
    from openai import OpenAI

    client = OpenAI(
        api_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    #model = "models/text-bison-001"
    model="gemini-2.0-flash",
    return client, model    

if __name__ == "__main__":
    import os
    os.system('clear')
    