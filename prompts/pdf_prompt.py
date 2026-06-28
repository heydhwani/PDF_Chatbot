from api import client
from rag.retriever import retrieve_context


def generate_answer(question):

    context = retrieve_context(question)

    prompt = f"""
You are a helpful PDF Chatbot.

Rules:
1. Answer ONLY using the given context.
2. Do not use outside knowledge.
3. If the answer is not available in the context, reply:
   "I couldn't find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    return response.text