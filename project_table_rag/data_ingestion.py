"""
1. Import Excel data as pandas DataFrame
2. Data Chunking
3. Data Embedding
"""

# package imports
import pandas as pd
from loguru import logger   
from langchain_core.documents import Document    
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# init global variables
LOG_LEVEL = "INFO"


# Set up logging
logger.add("project_table_rag/data_ingestion.log", rotation="1 MB", mode="w", level=LOG_LEVEL)

logger.info("Starting data ingestion process...")


# Load Excel data into a pandas DataFrame
payment_data = pd.read_excel("project_table_rag/data/emergency_payments_data.xlsx")
# Convert datetime to strings
payment_data["Created"] = payment_data["Created"].astype(str)
payment_data["Date created"] = payment_data["Date created"].astype(str) 
payment_data["Payment executed"] = payment_data["Payment executed"].astype(str)  
payment_data["Comments"] = payment_data["Comments"].astype(str)

# create row data as list of dicts, each dict is a row
row_data = payment_data.to_dict(orient="records")
logger.info(row_data[:5])  # Log first 5 rows for verification

# create chunks for embedding
chunks = [
    Document(
        page_content=", ".join([f"{k}: {v}" for k, v in row.items()]),
        metadata=row
    )
    for row in row_data
]
logger.info(f"Erste 3 Chunks: {chunks[:3]}")


# create embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# create and persist vector database
db = Chroma(embedding_function=embedding_model, persist_directory="project_table_rag/data/payment_data_db")

# add documents to the vector database
db.add_documents(documents=chunks)