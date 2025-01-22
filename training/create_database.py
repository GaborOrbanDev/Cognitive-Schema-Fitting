# %% import libraries
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import faiss
import openai
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import pandas as pd

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# %% create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# %% create the database
index = faiss.IndexFlatL2(len(embeddings.embed_query("sample text")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# %% reading and parsing measurements
df = pd.read_json("./training/measurements/all_aggregated_data.json")
measurements: list[Document] = []
ids: list[str] = []

for i, row in df.iterrows():
    measurement_id = row["measurement_id"]
    ids.append(measurement_id)
    measurements.append(
        Document(
            page_content=row["task"],
            metadata=row.drop(index=["task"]).to_dict()
        )
    )


# %% add documents to the vector store
vector_store.add_documents(measurements, ids=ids)

# %% save the vector store
vector_store.save_local("database", index_name="training_w_all_samples")