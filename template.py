import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec # Import Pinecone client and ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# No need to import Document explicitly if not directly instantiating it,
# as loaders and splitters handle it.

# -----------------------------
# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1") # Default to us-east-1 if not set

# Validate environment variables
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in environment variables. Please set it.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it.")

# -----------------------------
# Helper Functions (Improved from previous fix)

def load_pdf_file(data_path: str):
    """
    Loads PDF files from the specified directory.

    Args:
        data_path (str): The path to the directory containing PDF files.

    Returns:
        list: A list of loaded documents.
    """
    if not os.path.exists(data_path):
        print(f"Error: Directory '{data_path}' does not exist.")
        return []
    if not os.path.isdir(data_path):
        print(f"Error: Path '{data_path}' is not a directory.")
        return []

    try:
        print(f"Loading PDF files from: {data_path}")
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            print(f"No PDF documents found or loaded from '{data_path}'.")
        else:
            print(f"Successfully loaded {len(documents)} PDF documents.")
        return documents
    except Exception as e:
        print(f"An error occurred while loading PDF files: {e}")
        return []

def text_split(extracted_data: list):
    """
    Splits a list of documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        extracted_data (list): A list of Langchain Document objects.

    Returns:
        list: A list of text chunks (Langchain Document objects).
    """
    if not extracted_data:
        print("Warning: No data provided for text splitting. Returning empty list.")
        return []

    try:
        print(f"Splitting {len(extracted_data)} documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(extracted_data)
        print(f"Successfully split documents into {len(text_chunks)} chunks.")
        return text_chunks
    except Exception as e:
        print(f"An error occurred during text splitting: {e}")
        return []

def download_hugging_face_embeddings():
    """
    Downloads and initializes the HuggingFace embeddings model (all-MiniLM-L6-v2).

    Returns:
        HuggingFaceEmbeddings: An initialized HuggingFaceEmbeddings object.
    """
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    try:
        print(f"Initializing HuggingFace embeddings model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print("HuggingFace embeddings model initialized successfully.")
        return embeddings
    except Exception as e:
        print(f"An error occurred while initializing HuggingFace embeddings: {e}")
        return None

# -----------------------------
# Pinecone Initialization and Index Creation
index_name = "medicalbot"
embedding_dimension = 384 # This must match the dimension of 'all-MiniLM-L6-v2'

try:
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone client initialized.")

    # Check and create index
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
except Exception as e:
    print(f"Error during Pinecone initialization or index creation: {e}")
    exit() # Exit if Pinecone setup fails

# -----------------------------
# Step 1-3: Load and process documents
# Note: The path "E:/Downloads/New folder/Medical-Chatbot/Data" is an absolute Windows path.
# Consider using relative paths or os.path.join for better cross-platform compatibility.
data_directory = "E:/Downloads/New folder/Medical-Chatbot/Data"

docs = load_pdf_file(data_directory)
if not docs:
    print("Exiting: No documents loaded for processing.")
    exit()

chunks = text_split(docs)
if not chunks:
    print("Exiting: No text chunks generated.")
    exit()

embedding = download_hugging_face_embeddings()
if not embedding:
    print("Exiting: Failed to load embedding model.")
    exit()

# -----------------------------
# Upload to Pinecone (or connect to existing vector store)
try:
    print(f"Connecting to Pinecone vector store '{index_name}'...")
    # If the index is empty, from_documents will upload the chunks.
    # If the index has data, it will append or update based on document IDs.
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=index_name
    )
    print("✅ Vector store successfully connected/updated in Pinecone.")
except Exception as e:
    print(f"Error uploading/connecting to Pinecone vector store: {e}")
    exit()

# -----------------------------
# Prompt for Retrieval QA
prompt = ChatPromptTemplate.from_template("""
You are an assistant for answering medical-related questions.
Use the following context to answer the question.
If you don't know the answer, just say you don't know — don't make up an answer.
Keep it concise (2-3 sentences max).

Context:
{context}

Question:
{input}
""")

# -----------------------------
# LLM with Cohere
try:
    print("Initializing Cohere LLM (command-r)...")
    llm = ChatCohere(model="command-r", temperature=0.4, cohere_api_key=COHERE_API_KEY)
    print("Cohere LLM initialized.")
except Exception as e:
    print(f"Error initializing Cohere LLM: {e}")
    exit()

# -----------------------------
# Retrieval Chain
try:
    print("Setting up retrieval chain...")
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, prompt | llm) # Corrected chain creation
    print("Retrieval chain set up.")
except Exception as e:
    print(f"Error setting up retrieval chain: {e}")
    exit()

# -----------------------------
# Ask a question
print("\n--- Medical Chatbot Ready ---")
while True:
    user_question = input("Enter your medical question (or 'quit' to exit): ")
    if user_question.lower() == 'quit':
        print("Exiting chatbot. Goodbye!")
        break

    if not user_question.strip():
        print("Please enter a question.")
        continue

    try:
        print(f"Asking: '{user_question}'...")
        response = retrieval_chain.invoke({"input": user_question})
        print("\nAnswer:")
        print(response["answer"])
        if "context" in response:
            print("\n--- Retrieved Context (for debugging) ---")
            for doc in response["context"]:
                print(f"- Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
                print(f"  Snippet: {doc.page_content[:150]}...") # Print first 150 chars
            print("---------------------------------------")
        print("\n")
    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        print("Please try again or check your configuration.")