from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os # Import os for path checking and better error messages

# ✅ Extract data from PDF files in a directory
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
        # glob="*.pdf" ensures only PDF files are processed
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

# ✅ Split the documents into chunks (for embedding & vector search)
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

# ✅ Download HuggingFace embeddings (MiniLM-L6-v2 → 384 dimensions)
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
        # You can optionally test the embeddings to ensure they are working
        # Example: embeddings.embed_query("test query")
        print("HuggingFace embeddings model initialized successfully.")
        return embeddings
    except Exception as e:
        print(f"An error occurred while initializing HuggingFace embeddings: {e}")
        # Depending on the error, you might want to re-raise or return None
        return None # Return None if initialization fails to indicate an error