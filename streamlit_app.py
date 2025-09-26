import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set display config
st.set_page_config(
    page_title="AI Banking Assistant",
    page_icon="🏦",
    layout="wide"
)

# Datasets to be used
DATASET_PATHS = [
    "Bank_FAQs.csv",
    "synthetic_bank_queries_responses(Sheet1).csv",
    "synthetic_bank_customers(Sheet1).csv"
]

@st.cache_data
def load_all_data():
    loaded = {}
    for path in DATASET_PATHS:
        if os.path.exists(path):
            loaded[path] = pd.read_csv(path)
    return loaded

def combine_qa_sources(data_dict):
    qa_pairs = []
    for src, df in data_dict.items():
        cols = [c.lower() for c in df.columns]
        if "question" in cols and "answer" in cols:
            qcol = df.columns[cols.index("question")]
            acol = df.columns[cols.index("answer")]
            for _, row in df.iterrows():
                qa_pairs.append({"question": str(row[qcol]), "answer": str(row[acol]), "source": src})
        elif "query" in cols and "response" in cols:
            qcol = df.columns[cols.index("query")]
            acol = df.columns[cols.index("response")]
            for _, row in df.iterrows():
                qa_pairs.append({"question": str(row[qcol]), "answer": str(row[acol]), "source": src})
    return qa_pairs

@st.cache_resource
def initialize_chatbot(qa_pairs):
    questions = [item['question'].lower() for item in qa_pairs]
    answers = [item['answer'] for item in qa_pairs]
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=1200)
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors, answers, qa_pairs

def get_best_response(user_input, vectorizer, question_vectors, answers, qa_pairs):
    user_vector = vectorizer.transform([user_input.lower()])
    similarities = cosine_similarity(user_vector, question_vectors).flatten()
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    if best_sim > 0.3:
        return answers[best_idx]
    else:
        return "I'm not sure about that specific question. Please rephrase or ask something about accounts, loans, or banking services."

def main():
    st.title("🏦 AI Banking Assistant")
    st.write("Welcome to your intelligent banking assistant. Ask me anything about banking services, accounts, loans, or transactions.")

    # Load all datasets (silently)
    data_dict = load_all_data()

    # Combine QA sources for chatbot
    qa_pairs = combine_qa_sources(data_dict)
    
    # Add fallback Q&A if no data loaded
    if not qa_pairs:
        qa_pairs = [
            {"question": "How do I open a bank account?", "answer": "Visit your nearest branch with ID and address proof.", "source": "default"},
            {"question": "How to apply for a credit card?", "answer": "Fill out the form online or at any branch.", "source": "default"},
            {"question": "How do I check my balance?", "answer": "Use mobile app, ATM, or branch counter.", "source": "default"}
        ]

    # Initialize chatbot
    vectorizer, question_vectors, answers, qa_base = initialize_chatbot(qa_pairs)

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI banking assistant. How can I help you today?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your banking question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                response = get_best_response(prompt, vectorizer, question_vectors, answers, qa_base)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Optional: Add a sidebar with help
    with st.sidebar:
        st.header("💡 How to use")
        st.write("Simply type your banking questions in the chat box. I can help you with:")
        st.write("• Account information")
        st.write("• Loan inquiries")
        st.write("• Card services")
        st.write("• Banking procedures")
        st.write("• Transaction support")

if __name__ == "__main__":
    main()
