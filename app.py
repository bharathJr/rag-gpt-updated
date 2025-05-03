import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFMinerLoader, TextLoader, Docx2txtLoader

load_dotenv()
st.set_page_config(page_title="FSM Chatbot", layout="wide")

st.title("🧠 FSM Chatbot")
st.sidebar.header("📂 Upload FSM Files")

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True
)

documents = []

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".pdf"):
            loader = PDFMinerLoader(file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    if os.path.exists("vectorstore/index.faiss"):
        db = FAISS.load_local("vectorstore", embeddings)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local("vectorstore")
    st.sidebar.success("✅ Files uploaded and indexed!")

# Load embeddings and vectorstore
embeddings = OpenAIEmbeddings()
if os.path.exists("vectorstore/index.faiss"):
    db = FAISS.load_local("vectorstore", embeddings)
    retriever = db.as_retriever()
else:
    retriever = None
    st.warning("No documents uploaded or vectorstore found.")

# Chat Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("💬 Ask a question about FSM...")

if query and retriever:
    llm = ChatOpenAI(temperature=0)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    result = qa({"question": query, "chat_history": st.session_state.chat_history})

    st.session_state.chat_history.append((query, result["answer"]))

# Chat Display
if st.session_state.chat_history:
    for i, (user_q, bot_a) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**FSM GPT:** {bot_a}")
        st.markdown("---")
