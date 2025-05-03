import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
st.set_page_config(page_title="FSM Chatbot", layout="wide")

# --- UI ---
st.title("🧠 FSM Chatbot")
st.sidebar.header("📂 Upload FSM Files")

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# File uploader
uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# --- File Handling ---
if uploaded_files:
    documents = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        loader = UnstructuredFileLoader(file_path)
        documents.extend(loader.load())

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Load or create vector database
    embeddings = OpenAIEmbeddings()
    if os.path.exists("vectorstore.index"):
        # If the vector store exists, load it
        db = FAISS.load_local("vectorstore", embeddings)
        db.add_documents(chunks)  # Add new documents to the existing vector store
    else:
        # If no vector store exists, create one
        db = FAISS.from_documents(chunks, embeddings)

    # Save vector database
    db.save_local("vectorstore")
    st.sidebar.success("✅ Files uploaded and indexed!")

# --- Load Vector DB ---
db = FAISS.load_local("vectorstore", embeddings)
retriever = db.as_retriever()

# --- Chat Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat UI ---
query = st.text_input("💬 Ask a question about FSM...")

if query:
    llm = ChatOpenAI(temperature=0)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    result = qa({"question": query, "chat_history": st.session_state.chat_history})

    st.session_state.chat_history.append((query, result["answer"]))

# --- Show Chat History ---
if st.session_state.chat_history:
    for i, (user_q, bot_a) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**FSM GPT:** {bot_a}")
        st.markdown("---")
