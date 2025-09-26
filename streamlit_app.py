import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set display config
st.set_page_config(
    page_title="Multi-dataset Banking Chatbot",
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
        else:
            st.sidebar.warning(f"{path} not found")
    return loaded

def combine_qa_sources(data_dict):
    qa_pairs = []
    sources = []
    for src, df in data_dict.items():
        cols = [c.lower() for c in df.columns]
        if "question" in cols and "answer" in cols:
            qcol = df.columns[cols.index("question")]
            acol = df.columns[cols.index("answer")]
            for _, row in df.iterrows():
                qa_pairs.append({"question": str(row[qcol]), "answer": str(row[acol]), "source": src})
                sources.append(src)
        elif "query" in cols and "response" in cols:
            qcol = df.columns[cols.index("query")]
            acol = df.columns[cols.index("response")]
            for _, row in df.iterrows():
                qa_pairs.append({"question": str(row[qcol]), "answer": str(row[acol]), "source": src})
                sources.append(src)
    return qa_pairs, sources

@st.cache_resource
def initialize_chatbot(qa_pairs):
    questions = [item['question'].lower() for item in qa_pairs]
    answers = [item['answer'] for item in qa_pairs]
    # Vectorize questions for ML retrieval
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
        answer = answers[best_idx]
        src = qa_pairs[best_idx].get('source','unknown')
        return f"{answer} (Source: {os.path.basename(src)})"
    else:
        return "I'm not sure about that specific question. Please rephrase or ask something about accounts, loans, or banking services."

def main():
    st.title("🏦 Multi-dataset AI Banking Assistant")
    st.write("This app combines all approved banking Q&A and customer datasets for intelligent chatbot responses and insights.")

    # Load all datasets
    data_dict = load_all_data()

    # --- Sidebar with analytics ---
    st.sidebar.header("📚 Dataset Summary")
    for src, df in data_dict.items():
        st.sidebar.write(f"**{os.path.basename(src)}:** {len(df)} rows, {len(df.columns)} columns")
        st.sidebar.write(f"Columns: {list(df.columns)}")
        if "Account_Type" in df.columns:
            st.sidebar.write(f"Account types: {', '.join(df['Account_Type'].unique().astype(str)[:5])}")

    # Combine QA sources for chatbot
    qa_pairs, loaded_sources = combine_qa_sources(data_dict)
    if not qa_pairs:

        qa_pairs = [
            {"question": "How do I open a bank account?", "answer": "Visit your nearest branch with ID and address proof.", "source": "default"},
            {"question": "How to apply for a credit card?", "answer": "Fill out the form online or at any branch.", "source": "default"},
            {"question": "How do I check my balance?", "answer": "Use mobile app, ATM, or branch counter.", "source": "default"}
        ]

    st.sidebar.write(f"Total Q&A pairs loaded: {len(qa_pairs)}")

    # --- Chat Interface ---
    vectorizer, question_vectors, answers, qa_base = initialize_chatbot(qa_pairs)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask your banking question..."):
        st.session_state.messages.append({"role":"user", "content":prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_best_response(prompt, vectorizer, question_vectors, answers, qa_base)
                st.write(response)
                st.session_state.messages.append({"role":"assistant", "content":response})

    # --- Customer Data Analytics ---
    if "synthetic_bank_customers(Sheet1).csv" in data_dict:
        st.subheader("📊 Customer Data Overview")
        df = data_dict["synthetic_bank_customers(Sheet1).csv"]
        st.write("Sample Customers:")
        st.dataframe(df.head())
        if "Account_Type" in df.columns:
            st.write(df["Account_Type"].value_counts())

    # --- Q&A Sample Preview ---
    st.subheader("💡 Sample Q&A")
    for qa in qa_pairs[:5]:
        st.markdown(f"**Q:** {qa['question']}  \n**A:** {qa['answer']} *(Source: {os.path.basename(qa['source'])})*")

if __name__ == "__main__":
    main()
