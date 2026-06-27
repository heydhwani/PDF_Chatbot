import os
import pickle
import faiss
import numpy as np

from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from api import client

def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

def create_chunks(
    text,
    chunk_size=1000,
    chunk_overlap=200
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(text)

    return chunks

def get_embedding(chunk):

    response = client.models.embed_content(

        model="text-embedding-004",

        contents=chunk

    )

    embedding = response.embeddings[0].values

    return embedding

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
