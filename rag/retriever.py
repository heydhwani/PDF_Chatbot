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



def question_embedding(question):

    response = client.models.embed_content(

        model="text-embedding-004",

        contents=question

    )

    embedding = response.embeddings[0].values

    return np.array(
        [embedding],
        dtype=np.float32
    )



def retrieve_context(question, k=3):

    index, chunks = load_vectorstore()

    question_vector = question_embedding(question)

    distances, indices = index.search(
        question_vector,
        k
    )

    context = []

    for i in indices[0]:

        context.append(chunks[i])

    return "\n\n".join(context)