from openai import OpenAI
import os

def get_oaic_client(model_type, model):
    if model_type=='groq':        
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key= os.getenv('GROQ_API_KEY')
            )
        model = model if model else 'deepseek-r1-distill-llama-70b' 
        
    elif model_type=='ollama':
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama'
            ) # required, but unused
        model = model if model else 'deepseek-r1'

    elif model_type =='gemini':
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv('GEMINI_API_KEY'),
            )
        model = model if model else 'gemini-2.0-flash' 
        
    elif model_type=='mistral':
        client = OpenAI(
            base_url = "https://api.braintrust.dev/v1/proxy",
            api_key = os.getenv('MISTRAL_API_KEY')
            )
        model = model if model else 'mistral-large-latest'
       
    elif model_type=='openrouter':
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY')
            )
        model = "qwen/qwen2.5-vl-72b-instruct:free"
    
    return client, model
   
if __name__ == "__main__":
    import os
    os.system('clear')
    client, model = get_oaic_client('openrouter', None)

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
        