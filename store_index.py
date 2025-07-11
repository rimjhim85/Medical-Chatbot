import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
index_name = "medicalbot"

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Missing Pinecone credentials.")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
        print("Index created.")
    else:
        print("Index already exists.")
except Exception as e:
    print(f"Error setting up Pinecone index: {e}")
    exit()

try:
    print("Loading documents...")
    docs = load_pdf_file("Data/")
    if not docs:
        print("No documents found.")
        exit()
    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents...")
    chunks = text_split(docs)
    if not chunks:
        print("No chunks created.")
        exit()
    print(f"Created {len(chunks)} chunks.")

    embeddings = download_hugging_face_embeddings()
    if not embeddings:
        raise ValueError("Embeddings not loaded.")

    print("Uploading to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("Vector store uploaded to Pinecone.")
except Exception as e:
    print(f"Error during vector store creation: {e}")
