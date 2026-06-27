import pickle
import faiss
import numpy as np

from api import client

def load_vectorstore():

    index = faiss.read_index(
        "faiss_index/index.faiss"
    )

    with open(
        "faiss_index/chunks.pkl",
        "rb"
    ) as file:

        chunks = pickle.load(file)

    return index, chunks