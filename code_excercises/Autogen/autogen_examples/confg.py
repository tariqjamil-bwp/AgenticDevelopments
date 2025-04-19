import os
config_list = [
    {"api_type": "groq", "model": "llama-3.1-70b-versatile", "api_key": os.environ.get("GROQ_API_KEY")},
    {"api_type": "groq", "model": "llama3-70b-8192", "api_key": os.environ.get("GROQ_API_KEY")},
    {"api_type": "groq", "model": "llama3-groq-70b-8192-tool-use-preview", "api_key": os.environ.get("GROQ_API_KEY")},
    {"api_type": "groq", "model": "mixtral-8x7b-32768", "api_key": os.environ.get("GROQ_API_KEY")},
    {"api_type": "groq", "model": "llama3-groq-8b-8192-tool-use-preview", "api_key": os.environ.get("GROQ_API_KEY")},
    {"api_type": "groq", "model": "gemma2-9b-it", "api_key": os.environ.get("GROQ_API_KEY")},
    {"api_type": "groq", "model": "llama-3.1-8b-instant", "api_key": os.environ.get("GROQ_API_KEY")},
    {"api_type": "groq", "model": "llama-guard-3-8b", "api_key": os.environ.get("GROQ_API_KEY")},
    {"api_type": "groq", "model": "llama3-8b-8192", "api_key": os.environ.get("GROQ_API_KEY")},
    ]

llm_config={"config_list" : config_list}