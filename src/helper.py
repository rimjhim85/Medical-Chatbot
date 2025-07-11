import os
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf_file(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with open(pdf_path, "rb") as f:
                pdf_reader = PdfReader(f)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                documents.append(text)
    return documents

def text_split(documents):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents(documents)
    return chunks

def download_hugging_face_embeddings():
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Initializing HuggingFace embeddings model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        print(f"An error occurred while initializing HuggingFace embeddings: {e}")
        return None