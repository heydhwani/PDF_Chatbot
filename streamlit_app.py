import os
import streamlit as st

from rag.build_vectorstore import create_vectorstore
from prompts.pdf_prompt import generate_answer


st.set_page_config(page_title="PDF Chatbot")

st.title("📄 PDF Chatbot")


# Session State
if "database_created" not in st.session_state:
    st.session_state.database_created = False


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)


if uploaded_file is not None:

    # Create folder if not exists
    os.makedirs("PDF_documents", exist_ok=True)

    # Save uploaded PDF
    pdf_path = os.path.join(
        "PDF_documents",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())


    # Create Vector Store only once
    if not st.session_state.database_created:

        create_vectorstore(pdf_path)

        st.session_state.database_created = True

        st.success("PDF Processed Successfully ✅")


    # Question Input
    question = st.text_input("Ask your question")


    # Generate Answer
    if st.button("Get Answer"):

        if question.strip() == "":
            st.warning("Please enter a question.")

        else:

            answer = generate_answer(question)

            st.write("### Answer")

            st.write(answer)