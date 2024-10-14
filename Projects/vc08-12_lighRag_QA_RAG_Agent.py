# -*- coding: utf-8 -*-

import os
"""### Prepare Generator"""
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # or 'false'

# Importing required modules
import os
from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.model_client import ModelClient
from lightrag.components.model_client import OllamaClient

# Template for the QA generator
qa_template = r"""<SYS>
You are a helpful assistant.
</SYS>
User: {{input_str}}
You:"""

# Defining the SimpleQA component
class SimpleQA(Component):
    def __init__(self, model_client: ModelClient, model_kwargs: dict):
        super().__init__()
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=qa_template,
        )

    def call(self, input: dict) -> str:
        return self.generator.call({"input_str": str(input)})

    async def acall(self, input: dict) -> str:
        return await self.generator.acall({"input_str": str(input)})

"""### Prepare RAG"""

# Configuration for RAG components
configs = {
    "embedder": {
        "batch_size": 100,
        "model_kwargs": {
            "model": "thenlper/gte-base",
            "dimensions": 256,
            "encoding_format": "float",
        },
    },
    "retriever": {
        "top_k": 2,
    },
    "generator": {
         "model": "gemma:2b",
         "temperature": 0.3,
         "stream": True,
         "host": "http://localhost:11434"
    },
    
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 400,
        "chunk_overlap": 200,
    },
}

# Importing additional modules for RAG
from lightrag.core import Embedder, Sequential
from lightrag.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    TextSplitter,
)
from lightrag.core.types import Document, ModelClientType

# Preparing data pipeline
def prepare_data_pipeline():
    splitter = TextSplitter(**configs["text_splitter"])
    embedder = Embedder(
        model_client=ModelClientType.TRANSFORMERS(),
        model_kwargs={"model": "thenlper/gte-base"},
    )
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=configs["embedder"]["batch_size"]
    )
    data_transformer = Sequential(splitter, embedder_transformer)
    return data_transformer


# Function to read PDF and extract text
import PyPDF2
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


#Preparing demo documents
pdf_path1 = "data/21_patrick_2020_RAG.pdf" 
pdf_path2 = "data/AI for Everyone.pdf"

os.system('clear')
print('Ingesting Doc1')
doc1 = Document(
    meta_data={"title": "Linux for Beginers"},
    text=extract_text_from_pdf(pdf_path1),
    id="doc1",
)
print('Ingesting Doc2')
doc2 = Document(
    meta_data={"title": "AI for Everyone"},
    text=extract_text_from_pdf(pdf_path2),
    id="doc2",
)

print('Documents Ingestion complete')
# Preparing a local database to run the data pipeline to transform the documents and persist and retrieve document chunks with embeddings
from typing import List
from lightrag.core.db import LocalDB

def prepare_database_with_index(docs: List[Document], index_path: str = "index.faiss"):
    if os.path.exists(index_path):
        return None
    db = LocalDB()
    db.load(docs)
    data_transformer = prepare_data_pipeline()
    db.transform(data_transformer, key="data_transformer")
    db.save_state(index_path)
    print(db)

print('Preparing Database & Index')
# Getting the chunks with embeddings
prepare_database_with_index([doc1, doc2], index_path="index.faiss")

# Testing the database loading
db = LocalDB.load_state("index.faiss")
print(db)

# RAG task pipeline
from typing import Optional, Any
from lightrag.core.string_parser import JsonParser
from lightrag.components.retriever.faiss_retriever import FAISSRetriever

# Template for the RAG generator
rag_prompt_task_desc = r"""
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

Output JSON format:
{
    "answer": "The answer to the query",
}"""

# Defining the RAG component
class RAG(Component):
    def __init__(self, model_client: ModelClient, model_kwargs: dict, index_path: str = "index.faiss"):
        super().__init__()
        self.db = LocalDB.load_state(index_path)
        self.transformed_docs: List[Document] = self.db.get_transformed_data("data_transformer")
        embedder = Embedder(
            model_client=ModelClientType.TRANSFORMERS(),
            model_kwargs=configs["embedder"]["model_kwargs"],
        )
        self.retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )
        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)
        self.generator = Generator(
            prompt_kwargs={
                "task_desc_str": rag_prompt_task_desc,
            },
            model_client=model_client,
            model_kwargs=model_kwargs,
            output_processors=JsonParser(),
        )

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")
        prompt_kwargs = {
            "context_str": context,
            "input_str": query,
        }
        response = self.generator(prompt_kwargs=prompt_kwargs)
        return response

    def call(self, query: str) -> Any:
        retrieved_documents = self.retriever(query)
        for i, retriever_output in enumerate(retrieved_documents):
            retrieved_documents[i].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retriever_output.doc_indices
            ]
        print(f"retrieved_documents: \n {retrieved_documents}\n")
        context_str = self.retriever_output_processors(retrieved_documents)
        print(f"context_str: \n {context_str}\n")
        return self.generate(query, context=context_str), retrieved_documents

"""### Prepare ReAct Agent"""

# Importing required modules for ReAct Agent
from lightrag.components.agent import ReActAgent

# Defining tools for ReAct Agent
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b

async def add(a: int, b: int) -> int:
    """
    Add two numbers.
    """
    return a + b

def divide(a: float, b: float) -> float:
    """
    Divide two numbers.
    """
    return float(a) / b

# Testing ReAct Agent
def test_react_agent(model_client: ModelClient, model_kwargs: dict):
    tools = [multiply, add, divide]
    queries = [
        "What is the capital of France? and what is 465 times 321 then add 95297 and then divide by 13.2?",
        "Give me 5 words rhyming with cool, and make a 4-sentence poem using them",
    ]
    react = ReActAgent(
        max_steps=6,
        add_llm_as_fallback=True,
        tools=tools,
        model_client=model_client,
        model_kwargs=model_kwargs,
    )
    for query in queries:
        print(f"Query: {query}")
        agent_response = react.call(query)
        print(f"Agent response: {agent_response}")
        print("")

"""### Prepare all llama3 models"""

# Preparing and using the llama3 model with SimpleQA, RAG, and ReAct Agent

model = {
    "model_client": OllamaClient(),
    "model_kwargs": {"model": "gemma:2b"}
}

print('Output of Simple QA')
print('-------------------')
query = "Write down Linux composition"
print(f"Q: {query}")
#print(f"model: {model}")
qa = SimpleQA(**model)
print("REPLY:")
reply = qa(query)
print(reply.data)

# Initializing RAG and visualizing its structure

print('Output of RAG')
print('-------------')
print(f"model: {model}")
#query = "What is LINUX based on?"
print(f"Q: {query}")
rag = RAG(**model, index_path="index.faiss")
response, retrieved_documents = rag.call(query)
print(f"response: {response}")

# Using llama3.1 with ReAct Agent
print('Output of ReAct')
#print(f"model: {model}")
test_react_agent(**model)
