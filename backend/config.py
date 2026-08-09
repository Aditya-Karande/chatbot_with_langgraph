import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from langgraph.store.base import IndexConfig

load_dotenv()

# LLM
LLM = ChatGroq(model="openai/gpt-oss-120b",temperature=0)

# Postgres connection pool
DB_URL = os.environ["DATABASE_URL"]

connection_kwargs = {
    "autocommit":True,
    "prepare_threshold":0
}

pool = ConnectionPool(
    conninfo=DB_URL,
    max_size=20,
    kwargs=connection_kwargs
)

# Embeddings.
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=786,)

# short term memory: per-thread state, survives process restarts
checkpointer = PostgresSaver(pool)
checkpointer.setup()

# Long term memory: cross-thread facts, semantically searchable by user_id

store = PostgresStore(
    pool,
    index = IndexConfig(
        embed=embeddings,
        dims=786,
        fields=["content"]
    ),
)

store.setup()