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
from src.prompt import system_prompt
from src.helper import download_hugging_face_embeddings

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not PINECONE_API_KEY or not COHERE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Missing one or more environment variables.")

try:
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print("Pinecone client initialized.")

    print("Downloading Hugging Face embeddings...")
    embeddings = download_hugging_face_embeddings()
    if not embeddings:
        raise ValueError("Failed to load embedding model.")
    print("Embeddings loaded.")

    index_name = "medicalbot"
    print(f"Connecting to Pinecone index: {index_name}...")
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("Retriever ready.")

    print("Initializing Cohere LLM...")
    llm = ChatCohere(model="command-r", temperature=0.4, cohere_api_key=COHERE_API_KEY)
    print("Cohere LLM initialized.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    print("Prompt template ready.")

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("RAG chain ready.")

except Exception as e:
    print(f"Error during setup: {e}")
    exit()

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "I couldn't find an answer.")
        return str(answer)
    except Exception as e:
        print(f"Error in RAG chain: {e}")
        return "An error occurred. Try again.", 500

if __name__ == '__main__':
    print("Starting Flask app...")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)