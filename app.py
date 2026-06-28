from rag.build_vectorstore import create_vectorstore

from prompts.pdf_prompt import generate_answer


def chat_with_pdf(pdf_path, question):

    create_vectorstore(pdf_path)

    answer = generate_answer(question)

    return answer