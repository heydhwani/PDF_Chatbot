import os
import streamlit as st

from rag.build_vectorstore import create_vectorstore
from prompts.pdf_prompt import generate_answer


st.set_page_config(page_title="PDF Chatbot")

st.title("📄 PDF Chatbot")


# Session State
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = ""


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)


if uploaded_file is not None:

    # Create folder if it doesn't exist
    os.makedirs("PDF_documents", exist_ok=True)

    # Save uploaded PDF
    pdf_path = os.path.join(
        "PDF_documents",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())


    # Create Vector Store only if a new PDF is uploaded
    if uploaded_file.name != st.session_state.current_pdf:

        with st.spinner("Processing PDF..."):

            create_vectorstore(pdf_path)

        st.session_state.current_pdf = uploaded_file.name

        st.success("PDF processed successfully ✅")


    # Question Input
    question = st.text_input(
        "Ask a question"
    )


    # Get Answer
    if st.button("Get Answer"):

        if question.strip() == "":

            st.warning("Please enter a question.")

        else:

            with st.spinner("Generating Answer..."):

                answer = generate_answer(question)

            st.subheader("Answer")

            st.write(answer)