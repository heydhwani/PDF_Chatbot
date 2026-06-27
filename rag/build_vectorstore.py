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

def get_embedding(text):

    response = client.models.embed_content(

        model="text-embedding-004",

        contents=text

    )

    embedding = response.embeddings[0].values

    return embedding

