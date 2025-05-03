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
embeddings = OpenAIEmbeddings()
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
    if os.path.exists("vectorstore/index.faiss"):
        db = FAISS.load_local("vectorstore", embeddings)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    # Save vector database
    db.save_local("vectorstore")
    st.sidebar.success("✅ Files uploaded and indexed!")

# --- Load Vector DB ---
if os.path.exists("vectorstore/index.faiss"):
    db = FAISS.load_local("vectorstore", embeddings)
    retriever = db.as_retriever()
else:
    retriever = None
    st.warning("Upload and index documents to enable chat.")

# --- Chat Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat UI ---
query = st.text_input("💬 Ask a question about FSM...")

if query and retriever:
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
