import streamlit as st
import requests
import json
from datetime import datetime
import sqlite3
import bcrypt
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from streamlit_lottie import st_lottie
from streamlit_ace import st_ace
from streamlit_chat import message
from st_aggrid import AgGrid

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

def generate_response(prompt, exam_type, user_data):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    context = f"You are an AI tutor specialized in {exam_type} preparation. "
    context += "Previous interactions and user data:\n" + "\n".join(user_data)
    
    data = {
        "model": "togethercomputer/llama-2-70b-chat",
        "prompt": f"{context}\n\nUser: {prompt}\nAI:",
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('text', 'No response text found').strip()
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {str(http_err)}"
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {str(req_err)}"

def fine_tune_model(training_data):
    # Load and preprocess the data
    data = pd.read_csv(training_data)
    dataset = Dataset.from_pandas(data)

    # Initialize the tokenizer and model
    model_name = "facebook/llama-3b"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("path_to_save_your_model")
    tokenizer.save_pretrained("path_to_save_your_tokenizer")

    return "Model fine-tuned and saved successfully!"

# Streamlit UI
st.set_page_config(page_title="Exam Prep Assistant", layout="wide")

# Load Lottie animation
def load_lottie_url(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None

lottie_animation_url = "https://assets10.lottiefiles.com/packages/lf20_5nvbn5is.json"
lottie_animation = load_lottie_url(lottie_animation_url)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        color: #007bff;
        font-size: 36px;
        margin-bottom: 20px;
    }
    .header {
        color: #28a745;
        font-size: 28px;
        margin-top: 20px;
    }
    .input-container input, .input-container select {
        font-size: 18px;
        padding: 10px;
        margin: 5px 0;
    }
    .button-container button {
        background-color: #007bff;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .button-container button:hover {
        background-color: #0056b3;
    }
    .chat-response {
        padding: 10px;
        margin-top: 10px;
        border-radius: 5px;
    }
    .chat-response.user {
        background-color: #e9ecef;
        text-align: right;
    }
    .chat-response.ai {
        background-color: #f8f9fa;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'user' not in st.session_state:
    st.session_state.user = None

# Login/Register Page
if not st.session_state.user:
    st.markdown("<h1 class='title'>Welcome to Exam Prep Assistant</h1>", unsafe_allow_html=True)
    if lottie_animation:
        st_lottie(lottie_animation, height=300, width=300)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.markdown("<h2 class='header'>Login</h2>", unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            user = login_user(username, password)
            if user:
                st.session_state.user = user
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.markdown("<h2 class='header'>Register</h2>", unsafe_allow_html=True)
        new_username = st.text_input("New Username", key="register_username")
        new_password = st.text_input("New Password", type="password", key="register_password")
        exam_types = ["SAT", "ACT", "IELTS", "TOEFL"]
        selected_exam = st.selectbox("Select Exam Type", exam_types, key="exam_type")
        if st.button("Register", key="register_button"):
            register_user(new_username, new_password, selected_exam)
            st.success("Registered successfully! Please log in.")

# Main Application
else:
    st.markdown(f"<h1 class='title'>Welcome to your {st.session_state.user[3]} Prep Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Practice", "Resources", "Chat", "Fine-tune Model"])
    
    if page == "Dashboard":
        st.header("Your Progress")
        user_id = st.session_state.user[0]
        user_data = get_user_data(user_id)
        st.write("Recent Interactions:")
        for data in user_data[-5:]:  # Show last 5 interactions
            st.write(data)
    
    elif page == "Practice":
        st.header("Practice Section")
        st.write("Here you can practice questions specific to your exam type.")
        # Example practice questions; replace with dynamic content
        questions = [
            "Question 1: What is the capital of France?",
            "Question 2: What is the quadratic formula?",
            "Question 3: Describe the process of photosynthesis."
        ]
        st.selectbox("Choose a Practice Question", questions)

    elif page == "Resources":
        st.header("Study Resources")
        st.write("Here are some resources to help with your preparation:")
        
        # Example resources; replace with dynamic content
        resources = {
            "SAT": [
                {"title": "SAT Study Guide", "link": "https://example.com/sat-guide"},
                {"title": "SAT Practice Tests", "link": "https://example.com/sat-tests"}
            ],
            "ACT": [
                {"title": "ACT Prep Book", "link": "https://example.com/act-book"},
                {"title": "ACT Practice Questions", "link": "https://example.com/act-questions"}
            ],
            "IELTS": [
                {"title": "IELTS Preparation Course", "link": "https://example.com/ielts-course"},
                {"title": "IELTS Practice Tests", "link": "https://example.com/ielts-tests"}
            ],
            "TOEFL": [
                {"title": "TOEFL Study Material", "link": "https://example.com/toefl-material"},
                {"title": "TOEFL Practice Tests", "link": "https://example.com/toefl-tests"}
            ]
        }
        
        exam_resources = resources.get(st.session_state.user[3], [])
        for resource in exam_resources:
            st.markdown(f"- [{resource['title']}]({resource['link']})")

    elif page == "Chat":
        st.header("Chat with AI Tutor")
        user_input = st.text_input("Ask a question:")
        if st.button("Submit"):
            user_data = get_user_data(st.session_state.user[0])
            response = generate_response(user_input, st.session_state.user[3], user_data)
            st.markdown("<div class='chat-response user'>User: {}</div>".format(user_input), unsafe_allow_html=True)
            st.markdown("<div class='chat-response ai'>AI: {}</div>".format(response), unsafe_allow_html=True)
            save_user_data(st.session_state.user[0], f"User: {user_input}\nAI: {response}")

    elif page == "Fine-tune Model":
        st.header("Fine-tune the AI Model")
        training_data = st.file_uploader("Upload Training Data (CSV)")
        if training_data is not None:
            if st.button("Fine-tune Model"):
                result = fine_tune_model(training_data)
                st.success(result)
    
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.experimental_rerun()
