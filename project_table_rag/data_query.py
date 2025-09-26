# packages
import os
from loguru import logger
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# init global variables
LOG_LEVEL = "INFO"
MODEL_NAME = "x-ai/grok-4-fast"  # openai/gpt-oss-120b # openai/gpt-5
API_KEY = os.getenv("OPENROUTER_API_KEY")   # openrouter.ai


# Set up logging
logger.add("project_table_rag/data_query.log", rotation="1 MB", mode="w", level=LOG_LEVEL)

logger.info("Starting data query...")

# init embedding model and db
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(embedding_function=embedding_model, persist_directory="project_table_rag/data/payment_data_db")

# setup retriever
retriever = db.as_retriever( 
    search_kwargs={
        "k": 20,
        "score_threshold": 0.0,     # higher means more similar
    },
    search_type="similarity_score_threshold"
)


def table_query(user_query: str):
    """ Query the vector DB and return results
    input: user query
    returns: answer from AI model
    """

    # --------- Retrieval from vector DB -------------
    context = "; ".join(reply.page_content for reply in retriever.invoke(user_query))
    logger.info(f"Context from DB: {context}")

    # --------- augmenting user query with context -------------
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an accountant and finance expert who answers questions based solely on the provided context {context}. The context is based on table. People mostly care when an invoice was paid. If the context does not allow answering the question, respond with 'I don't know'."),
        ("user", "User question: {user_query}; Context: {context}"),
    ])

    # --------- Generating answer -------------
    model = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    chain = LLMChain(prompt=prompt_template, llm=model, output_parser=StrOutputParser())

    response = chain.run({
        "user_query": user_query,
        "context": context
    })

    logger.info(f"Response: {response}")
    return response


# ---------- Query examples -------------
# "Which payments were executed in September 2025? List all of them"
# "Is Status of TASK0872290 complete?"
# "When was the last payment executed to payment recipient Fraport AG?"

query = "Which payments were executed in September 2025?"


print(table_query(query))
