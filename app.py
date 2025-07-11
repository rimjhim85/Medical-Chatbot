from flask import Flask, render_template, jsonify, request
import os
from dotenv import load_dotenv
from pinecone import Pinecone # Import Pinecone client for explicit initialization
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Import your custom prompt
from src.prompt import system_prompt
# Import your helper functions
from src.helper import download_hugging_face_embeddings

app = Flask(__name__)

# -----------------------------
# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1") # Ensure environment is loaded
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Validate environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it.")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in environment variables. Please set it.")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not found in environment variables. Please set it (e.g., us-east-1).")


# -----------------------------
# Initialize Pinecone client (needed for Langchain's from_existing_index)
try:
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT) # Pass environment here
    print("Pinecone client initialized.")
except Exception as e:
    print(f"Error initializing Pinecone client: {e}")
    exit() # Exit if Pinecone client cannot be initialized

# -----------------------------
# Load embeddings and connect to Pinecone
try:
    print("Downloading Hugging Face embeddings...")
    embeddings = download_hugging_face_embeddings()
    if not embeddings:
        raise ValueError("Failed to download or load embedding model.")
    print("Embeddings loaded.")

    index_name = "medicalbot"
    print(f"Connecting to existing Pinecone index: {index_name}...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print("Pinecone vector store connected.")

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("Retriever created.")
except Exception as e:
    print(f"Error setting up Pinecone vector store or retriever: {e}")
    exit() # Exit if vector store setup fails

# -----------------------------
# Use Cohere model
try:
    print("Initializing Cohere LLM (command-r)...")
    llm = ChatCohere(model="command-r", temperature=0.4, cohere_api_key=COHERE_API_KEY)
    print("Cohere LLM initialized.")
except Exception as e:
    print(f"Error initializing Cohere LLM: {e}")
    exit() # Exit if LLM cannot be initialized

# -----------------------------
# Prompt setup
# The system_prompt is imported from src.prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
print("Chat prompt template created.")

# -----------------------------
# Create RAG chain
try:
    print("Creating RAG chain components...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("RAG chain created successfully.")
except Exception as e:
    print(f"Error creating RAG chain: {e}")
    exit() # Exit if chain creation fails

# -----------------------------
# Flask Routes
@app.route("/")
def index():
    """Renders the main chat interface."""
    return render_template('chat.html')

@app.route("/get", methods=["POST"]) # Changed to POST as data is sent in form
def chat():
    """Handles chat messages from the frontend and returns bot responses."""
    if request.method == "POST":
        msg = request.form.get("msg") # Use .get() for safer access
        if not msg:
            return jsonify({"error": "No message provided"}), 400

        print(f"User Question: {msg}")
        try:
            response = rag_chain.invoke({"input": msg})
            answer = response.get("answer", "I couldn't find an answer for that.")
            print(f"Response: {answer}")
            return str(answer) # Return as string for simple text response
        except Exception as e:
            print(f"Error during RAG chain invocation: {e}")
            return "An error occurred while processing your request. Please try again.", 500
    return "Method Not Allowed", 405 # Should not be reached with methods=["POST"]

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=8080, debug=True)