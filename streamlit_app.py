import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Banking Assistant", page_icon="🏦", layout="wide")

@st.cache_data
def load_data():
    # One dataset, Bank_FAQs.csv with 'question' and 'answer' columns
    if "Bank_FAQs.csv" in os.listdir():
        df = pd.read_csv("Bank_FAQs.csv")
        df = df[df['question'].notna() & df['answer'].notna()]
        return [{"q": str(row['question']), "a": str(row['answer'])} for _, row in df.iterrows()]
    # If missing, use default
    return [
        {"q": "How to open a savings account?", "a": "Visit any bank with your ID and address proof."},
        {"q": "Current home loan rates?", "a": "Home loan interest rates typically range from 8.5% to 12%. For details, check with your bank."}
    ]

@st.cache_resource
def build_indexes(qa_pairs):
    # Sentence transformers for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    questions = [qa['q'] for qa in qa_pairs]
    answers = [qa['a'] for qa in qa_pairs]
    emb = model.encode(questions)
    vec = TfidfVectorizer(stop_words='english', max_features=1200)
    tfidf = vec.fit_transform([q.lower() for q in questions])
    return {"model": model, "questions": questions, "answers": answers, "emb": emb, "tfidf": tfidf, "vectorizer": vec}

def hybrid_search(query, idxs, topk=1):
    # Embedding similarity first
    query_emb = idxs['model'].encode([query])
    emb_sims = np.dot(query_emb, idxs['emb'].T)[0]
    top = np.argsort(emb_sims)[-topk:][::-1]
    if emb_sims[top[0]] > 0.7:
        return idxs['answers'][top[0]]
    # TF-IDF fallback
    query_vec = idxs['vectorizer'].transform([query.lower()])
    tfidf_sims = cosine_similarity(query_vec, idxs['tfidf']).flatten()
    best = np.argmax(tfidf_sims)
    if tfidf_sims[best] > 0.3:
        return idxs['answers'][best]
    # Fallback
    return "I can help you with banking, accounts, cards, loans, and transfers. Please provide more details."

def main():
    st.title("🏦 AI Banking Assistant")
    qa_pairs = load_data()
    idxs = build_indexes(qa_pairs)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today with banking?"}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    if prompt := st.chat_input("Ask me about banking..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            response = hybrid_search(prompt, idxs)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
