import streamlit as st
import requests
import json
import sqlite3
import bcrypt
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, pipeline
import faiss
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
API_URL = "https://api.together.ai/v1/completions"
API_KEY = "5f2933c4375bb56d19475dc1088c5c01f140106f1754faebd6fed6e4bee53adc"

# Database setup
conn = sqlite3.connect('user_data.db')
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY, username TEXT, password TEXT, exam_type TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS user_data
             (id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT, timestamp TEXT)''')

conn.commit()

# Embedding model setup
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# FAISS index setup
embedding_dim = model.config.hidden_size
index = faiss.IndexFlatL2(embedding_dim)

# Helper functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

def register_user(username, password, exam_type):
    hashed_password = hash_password(password)
    c.execute("INSERT INTO users (username, password, exam_type) VALUES (?, ?, ?)",
              (username, hashed_password, exam_type))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    if user and verify_password(user[2], password):
        return user
    return None

def save_user_data(user_id, content):
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO user_data (user_id, content, timestamp) VALUES (?, ?, ?)",
              (user_id, content, timestamp))
    conn.commit()

def get_user_data(user_id):
    c.execute("SELECT content FROM user_data WHERE user_id=?", (user_id,))
    return [row[0] for row in c.fetchall()]

# RAG Functions
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def embed_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def add_to_faiss_index(texts):
    for text in texts:
        embedding = embed_text(text)
        index.add(embedding)

def search_faiss_index(query, k=5):
    query_embedding = embed_text(query)
    _, indices = index.search(query_embedding, k)
    return indices.flatten()

def generate_response_with_rag(prompt, exam_type, user_data):
    relevant_indices = search_faiss_index(prompt)
    relevant_texts = [user_data[i] for i in relevant_indices]

    context = f"You are an AI tutor specialized in {exam_type} preparation."
    context += " Previous interactions and user data:\n" + "\n".join(relevant_texts)

    data = {
        "model": "togethercomputer/llama-2-70b-chat",
        "prompt": f"{context}\n\nUser: {prompt}\nAI:",
        "max_tokens": 500,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('text', 'No response text found').strip()
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {str(http_err)}"
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {str(req_err)}"

# Streamlit UI
st.set_page_config(page_title="Exam Prep Assistant", layout="wide")

st.title("RAG-Enhanced Exam Preparation Chatbot")

# Sidebar for document upload
st.sidebar.header("Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        save_user_data(1, text)  # Save to DB (Example User ID = 1)
        add_to_faiss_index([text])
    st.sidebar.success("Files uploaded and indexed successfully!")

# Main Chat Section
if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    st.write("Please login or register to start using the chatbot.")
else:
    st.write(f"Welcome back, {st.session_state.user[1]}!")
    prompt = st.text_input("Ask a question:")
    if st.button("Submit"):
        user_data = get_user_data(st.session_state.user[0])
        response = generate_response_with_rag(prompt, st.session_state.user[3], user_data)
        st.write(response)
