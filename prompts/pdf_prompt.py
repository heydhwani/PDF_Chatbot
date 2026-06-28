from api import client
from rag.retriever import retrieve_context


def generate_answer(question):

    context = retrieve_context(question)

    prompt = f"""

You are an intelligent PDF Question Answering Assistant.

Your task is to answer the user's question ONLY using the retrieved context from the uploaded PDF.

Instructions:
1. Read the retrieved context carefully before answering.
2. If the answer exists in the context, answer accurately and concisely.
3. Do NOT use your own knowledge.
4. Do NOT make assumptions or guess.
5. If the answer is partially available, answer using only the available information.
6. If the answer is completely missing from the context, reply exactly with:
   "I couldn't find the answer in the uploaded PDF."

Retrieved Context:
------------------
{context}

User Question:
--------------
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt
    )

    return response.text