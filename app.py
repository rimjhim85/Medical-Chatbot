from flask import Flask, render_template, jsonify, request
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Local modules
from src.prompt import system_prompt
from src.helper import download_hugging_face_embeddings

# Flask App
app = Flask(__name__)

# -------------------- Load Environment Variables --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Missing Pinecone credentials.")
if not COHERE_API_KEY:
    raise ValueError("Missing Cohere API key.")

# -------------------- Initialize Pinecone --------------------
try:
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print("Pinecone client initialized.")
except Exception as e:
    print(f"Failed to initialize Pinecone: {e}")
    exit()

# -------------------- Load Embeddings --------------------
try:
    print("Loading Hugging Face embeddings...")
    embeddings = download_hugging_face_embeddings()
    if not embeddings:
        raise Exception("Embedding load failed.")
except Exception as e:
    print(f"Embedding Error: {e}")
    exit()

# -------------------- Connect to Pinecone Index --------------------
try:
    index_name = "medicalbot"
    print(f"Connecting to Pinecone index: {index_name}")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
except Exception as e:
    print(f"Pinecone Vector Store Error: {e}")
    exit()

# -------------------- Initialize Cohere LLM --------------------
try:
    print("Initializing Cohere LLM...")
    llm = ChatCohere(model="command-r", temperature=0.4, cohere_api_key=COHERE_API_KEY)
except Exception as e:
    print(f"Cohere Initialization Error: {e}")
    exit()

# -------------------- Prompt Template --------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# -------------------- RAG Chain --------------------
try:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
except Exception as e:
    print(f"Error creating RAG chain: {e}")
    exit()

# -------------------- Flask Routes --------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "Message missing"}), 400
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        return str(answer)
    except Exception as e:
        print(f"RAG Error: {e}")
        return "Something went wrong. Please try again.", 500

# -------------------- App Entry --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
