from api import client
from rag.retriever import retrieve_context


def generate_answer(question):

    context = retrieve_context(question)

    prompt = f"""

You are a PDF assistant.

Answer ONLY from the retrieved context.

If the answer is present in the context,
answer it exactly.

Do not guess.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt
    )

    return response.text