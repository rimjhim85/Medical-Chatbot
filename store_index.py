import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# It's generally recommended to import specific classes/functions from langchain
# rather than relying on `langchain_community` directly if possible,
# though for Pinecone integration, it might be the standard.
from langchain_pinecone import PineconeVectorStore # Corrected import for Langchain Pinecone

# Assuming src.helper contains these functions
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings 

# ✅ Step 1: Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# It's good practice to also get the environment variable for region if it might change
# or to make it explicit that it's hardcoded.
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1") # Get from env or default
index_name = "medicalbot"

# Validate environment variables early
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not found or defaulted incorrectly.")

# ✅ Step 2: Initialize Pinecone using the latest client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Step 3: Create index if it doesn't exist
# Check for index existence and configuration
try:
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384,  # Ensure this matches your embedding model's dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
        # Optional: You might want to check if the existing index configuration matches
        # e.g., dimension, metric, and spec. If not, you might need to re-create it
        # (which would delete existing data) or handle it differently.
        # For this example, we'll assume it's correctly configured if it exists.
except Exception as e:
    print(f"Error checking or creating Pinecone index: {e}")
    # Depending on the application, you might want to exit or handle more gracefully
    exit() # Exit if index creation fails as subsequent steps will fail


# ✅ Step 4: Load, split, and embed documents
try:
    print("Loading PDF files...")
    docs = load_pdf_file("Data/")
    if not docs:
        print("No documents loaded from 'Data/' directory. Please ensure files exist.")
        # Decide if you want to proceed without documents or exit
        exit() # Exit if no documents to process

    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents into chunks...")
    chunks = text_split(docs)
    if not chunks:
        print("No chunks generated from documents. Check text_split function.")
        exit()

    print(f"Generated {len(chunks)} chunks.")

    print("Downloading Hugging Face embeddings model...")
    embeddings = download_hugging_face_embeddings()
    if not embeddings:
        raise ValueError("Failed to download or load embedding model.")
    print("Embedding model loaded.")

except Exception as e:
    print(f"Error during document loading, splitting, or embedding: {e}")
    exit()

# ✅ Step 5: Upload chunks to Pinecone
try:
    print(f"Uploading {len(chunks)} chunks to Pinecone index '{index_name}'...")
    vectorstore = PineconeVectorStore.from_documents( # Changed to PineconeVectorStore
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("✅ Vector store successfully created/updated in Pinecone.")

except Exception as e:
    print(f"Error uploading chunks to Pinecone: {e}")