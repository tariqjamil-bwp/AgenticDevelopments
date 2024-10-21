#- **Tutorial Title:** Build Your Own Auto-GPT Apps with LangChain (Python Tutorial)
#- **Tutorial Author:** Dave Ebbelaar
#- **Tutorial Link:** [Link to Tutorial Video](https://www.youtube.com/watch?v=NYSWn1ipbgg)
#- **Tutorial GitHub Link:** [Link to GitHub Tutorial](https://github.com/daveebbelaar/langchain-experiments/tree/main/youtube)


from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def create_db_from_youtube_video_url(youtube_url: str):
    loader = YoutubeLoader.from_youtube_url(youtube_url=youtube_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents=transcript)

    db = FAISS.from_documents(documents=docs, embedding=embeddings)

    return db

# create_db_from_youtube_video_url(youtube_url=youtube_url)


def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    documents = db.similarity_search(query=query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in documents])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {documents}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(template=human_template)

    # Chat prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        messages=[system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, documents=docs_page_content)
    response = response.replace("\n", "")

    return response, documents


# Example usage:
youtube_url = 'https://www.youtube.com/watch?v=rz-B9DKk9m4'
db = create_db_from_youtube_video_url(youtube_url=youtube_url)

query = "What is a summary of what Andy is saying?"

response, docs = get_response_from_query(db=db, query=query, k=4)
print(textwrap.fill(response, width=85))

# Printing out the docs allows for fact checking by the end user as they can see the context of the answer
print(docs)
