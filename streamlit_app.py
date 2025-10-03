import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Banking Assistant", page_icon="🏦", layout="wide")

@st.cache_data
def load_data():
    filename = "Bank_FAQs.csv"
    if not os.path.exists(filename) and os.path.exists(os.path.join("app", filename)):
        filename = os.path.join("app", filename)

    df = pd.read_csv(filename)

    # Detect columns for questions and answers
    question_cols = [c for c in df.columns if 'question' in c.lower() or 'query' in c.lower()]
    answer_cols = [c for c in df.columns if 'answer' in c.lower() or 'response' in c.lower()]

    if not question_cols or not answer_cols:
        st.error("Cannot find question or answer columns in dataset.")
        return []

    q_col = question_cols[0]
    a_col = answer_cols[0]

    df = df[df[q_col].notna() & df[a_col].notna()]

    blacklist = ["customer_", "dear customer", "your loan of", "rs.", "$", "icici", "hdfc", "axis", "sbi"]
    qa_pairs = []
    for _, row in df.iterrows():
        q, a = str(row[q_col]).strip(), str(row[a_col]).strip()
        if len(q) > 5 and len(a) > 10 and not any(bad in a.lower() for bad in blacklist):
            qa_pairs.append({"question": q, "answer": make_generic_answer(a), "category": categorize_question(q)})

    qa_pairs += generic_banking_qa()
    return qa_pairs

def categorize_question(question):
    q = question.lower()
    if "card" in q: return "cards"
    if any(x in q for x in ["loan", "emi", "credit"]): return "loans"
    if any(x in q for x in ["account", "balance", "saving"]): return "accounts"
    if any(x in q for x in ["transfer", "payment", "neft", "rtgs", "upi"]): return "transfers"
    return "general"

def make_generic_answer(a):
    replacements = {
        "HDFC Bank": "your bank",
        "HDFC": "your bank",
        "ICICI Bank": "your bank",
        "SBI": "your bank",
        "Axis Bank": "your bank",
        "NetBanking login": "internet banking",
        "ForexPlus Chip card": "prepaid cards",
        "our branch": "any bank branch",
        "our bank": "your bank",
        "we offer": "banks typically offer"
    }
    for k, v in replacements.items():
        a = a.replace(k, v)
    return a

def generic_banking_qa():
    return [
        {"question": "How to open a savings account?", "answer": "Visit any bank branch with valid ID and address proof to open your savings account.", "category": "accounts"},
        {"question": "What is home loan interest rate?", "answer": "Home loan interest rates typically range between 8% to 12% per annum. Please check with your bank for current rates.", "category": "loans"},
        {"question": "Which documents are required for credit card?", "answer": "Banks usually require identity proof, address proof, income proof and photograph for credit card applications.", "category": "cards"},
        {"question": "How to transfer money online?", "answer": "You can transfer money online using NEFT, RTGS, IMPS or UPI via internet or mobile banking.", "category": "transfers"},
        {"question": "What are personal loan interest rates?", "answer": "Personal loan interest rates usually vary from 10% to 24% per annum depending on your credit score and bank policies.", "category": "loans"}
    ]

@st.cache_resource
def build_indexes(qa_pairs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    questions = [item["question"] for item in qa_pairs]
    answers = [item["answer"] for item in qa_pairs]
    embeddings = model.encode(questions, show_progress_bar=False)
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    faiss_index = faiss.IndexFlatIP(norm_embeddings.shape[1])
    faiss_index.add(norm_embeddings.astype('float32'))
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=2000)
    tfidf_matrix = vectorizer.fit_transform([q.lower() for q in questions])
    return {
        "model": model,
        "faiss_index": faiss_index,
        "embeddings": norm_embeddings,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "questions": questions,
        "answers": answers,
        "qa_pairs": qa_pairs
    }

def hybrid_search(query, index_data, top_k=3):
    query_emb = index_data["model"].encode([query])
    query_emb = query_emb / np.linalg.norm(query_emb)
    distances, indices = index_data["faiss_index"].search(query_emb.astype('float32'), top_k)
    if distances[0][0] > 0.7:
        return index_data["answers"][indices[0][0]]
    query_vec = index_data["vectorizer"].transform([query.lower()])
    sim_scores = cosine_similarity(query_vec, index_data["tfidf_matrix"]).flatten()
    best_idx = sim_scores.argmax()
    if sim_scores[best_idx] > 0.35:
        return index_data["answers"][best_idx]
    return fallback_response(query)

def fallback_response(query):
    ql = query.lower()
    if "loan" in ql:
        return "Banks offer various loans including personal, home, and business loans. Please specify the loan type for more info."
    if "card" in ql:
        return "I can help you with credit and debit card queries. Please provide more details."
    if "account" in ql:
        return "You can open savings or current accounts by visiting any bank branch with required documents."
    if "transfer" in ql or "payment" in ql:
        return "Money transfers can be done via NEFT, RTGS, IMPS, or UPI using internet or mobile banking."
    return "Please specify your banking need like accounts, loans, cards, transfers, etc."

def main():
    st.title("🏦 AI Banking Assistant")
    qa_pairs = load_data()
    index_data = build_indexes(qa_pairs)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you with your banking needs today?"}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if prompt := st.chat_input("Ask your banking question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            response = hybrid_search(prompt, index_data)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
