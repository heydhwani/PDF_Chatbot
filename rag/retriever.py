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

        model="gemini-embedding-001",

        contents=question

    )

    embedding = response.embeddings[0].values

    question_vector = np.array(
        [embedding],
        dtype=np.float32
    )

    faiss.normalize_L2(question_vector)

    return question_vector


def retrieve_context(question, k=11):

    index, chunks = load_vectorstore()

    question_vector = question_embedding(question)

    distances, indices = index.search(
    question_vector,
    k
)

    print("=" * 80)
    print("Indices:", indices)
    print("Scores:", distances)

    context = []

    for i in indices[0]:

        print("\n----------------------")
        print(f"Chunk Index: {i}")
        print(chunks[i])

        context.append(chunks[i])

    print("=" * 80)

    return "\n\n".join(context)