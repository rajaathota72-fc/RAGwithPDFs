import fitz  # PyMuPDF
import os
import sqlite3
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from config import MODEL_CONFIGS, OPENAI_API_KEY

load_dotenv()

# Define a class to encapsulate document sections
class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata else {}

# Step 1: Upload PDF and Extract Text
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text_into_documents(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_text(text)
    documents = [Document(split) for split in splits]
    return documents

# Save and load vectorstore using SQLite
def save_vectorstore_sqlite(documents, embeddings, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents (content TEXT, metadata TEXT)''')
    c.execute('''DELETE FROM documents''')

    for doc in documents:
        metadata_json = json.dumps(doc.metadata)  # Convert metadata to JSON string
        c.execute("INSERT INTO documents (content, metadata) VALUES (?, ?)", (doc.page_content, metadata_json))

    conn.commit()
    conn.close()

def load_vectorstore_sqlite(db_path):
    if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT content, metadata FROM documents")
            rows = c.fetchall()
            conn.close()
            documents = [Document(content=row[0], metadata=json.loads(row[1])) for row in rows]  # Convert metadata back to dict
            return documents
        except (EOFError, sqlite3.DatabaseError):
            print(f"Warning: {db_path} is empty or corrupted. Creating a new vectorstore.")
            return None
    return None

# Step 2: Functions for Implementing RAG using Vector Store
def initialize_rag_model(documents, embeddings, model_name, db_path):
    loaded_documents = load_vectorstore_sqlite(db_path)
    if loaded_documents is None:
        save_vectorstore_sqlite(documents, embeddings, db_path)
        loaded_documents = documents

    vectorstore = Chroma.from_documents(documents=loaded_documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    model_config = MODEL_CONFIGS[model_name]
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model_config["model"], temperature=model_config["temperature"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Step 3: Price Calculation
def calculate_rag_price(rag_chain, user_input, model_name):
    model_config = MODEL_CONFIGS[model_name]
    with get_openai_callback() as cb_rag:
        rag_response = rag_chain.invoke(user_input)
        rag_cost = cb_rag.total_cost
    return rag_response, rag_cost
