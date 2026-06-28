import os
import pickle
import faiss
import numpy as np

from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from api import client


#READING PDF

def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


#CREATING CHUNKS

def create_chunks(
    text,
    chunk_size=500,
    chunk_overlap=100
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(text)

    return chunks

def get_embedding(chunk):

    response = client.models.embed_content(

        model="gemini-embedding-001",

        contents=chunk

    )

    embedding = response.embeddings[0].values

    return embedding

#FAISS DATABASE

def build_faiss(chunks):

    embeddings = []

    for chunk in chunks:

        embedding = get_embedding(chunk)

        embeddings.append(embedding)

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index

#saving database

def save_vectorstore(index, chunks):

    os.makedirs(
        "faiss_index",
        exist_ok=True
    )

    faiss.write_index(
        index,
        "faiss_index/index.faiss"
    )

    with open(
        "faiss_index/chunks.pkl",
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

#calling functions

def create_vectorstore(pdf_path):

    text = extract_text(pdf_path)

    chunks = create_chunks(text)

    index = build_faiss(chunks)

    save_vectorstore(index, chunks)

    return "Vector Store Created Successfully"