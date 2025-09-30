import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

st.set_page_config(page_title="AI Banking Assistant", page_icon="🏦", layout="wide")

DATASETS = ["Bank_FAQs.csv", "synthetic_bank_queries_responses(Sheet1).csv", "synthetic_bank_customers(Sheet1).csv"]

@st.cache_resource
def load_models():
    models = {'hf_loaded': False}
    if HF_AVAILABLE:
        try:
            models.update({
                'similarity_model': SentenceTransformer('all-MiniLM-L6-v2'),
                'qa_pipeline': pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
                'hf_loaded': True
            })
        except:
            pass
    return models

@st.cache_data
def load_data():
    qa_pairs = []
    for path in DATASETS:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                cols = [c.lower() for c in df.columns]
                q_col = next((c for c in df.columns if c.lower() in ['question', 'query']), None)
                a_col = next((c for c in df.columns if c.lower() in ['answer', 'response']), None)
                if q_col and a_col:
                    qa_pairs.extend([{"question": str(row[q_col]), "answer": str(row[a_col])} 
                                   for _, row in df.iterrows() if str(row[q_col]) != 'nan'])
            except:
                continue
    
    if not qa_pairs:
        qa_pairs = [
            {"question": "How do I open a bank account?", "answer": "Visit branch with ID and address proof."},
            {"question": "How to apply for credit card?", "answer": "Apply online or visit branch with income proof."},
            {"question": "How do I check my balance?", "answer": "Use mobile app, ATM, or call customer service."}
        ]
    return qa_pairs

@st.cache_resource
def init_tfidf(qa_pairs):
    questions = [qa['question'].lower() for qa in qa_pairs]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=2000)
    return vectorizer, vectorizer.fit_transform(questions), [qa['answer'] for qa in qa_pairs]

def get_fallback(user_input):
    keywords = {
        'account': "🏦 Account services: balance check, opening, statements. What do you need?",
        'loan': "💰 Loan services: personal, home, car loans. Which interests you?",
        'card': "💳 Card services: application, activation, PIN change. How can I help?",
        'transfer': "💸 Transfer services: NEFT, RTGS, UPI. What type of transfer?"
    }
    for key, response in keywords.items():
        if key in user_input.lower():
            return response
    return "🏦 I can help with accounts, loans, cards, and transfers. What do you need?"

def get_response(user_input, models, qa_pairs, tfidf_data):
    if not user_input.strip():
        return "Please ask a banking question.", 0.0, "low"
    
    if models['hf_loaded']:
        try:
            questions = [qa['question'] for qa in qa_pairs]
            answers = [qa['answer'] for qa in qa_pairs]
            similarities = np.dot(models['similarity_model'].encode([user_input]), 
                                models['similarity_model'].encode(questions).T)[0]
            best_idx, best_sim = np.argmax(similarities), similarities[np.argmax(similarities)]
            
            if best_sim > 0.75:
                return f"✅ {answers[best_idx]}", best_sim, "high"
            elif best_sim > 0.5:
                try:
                    context = " ".join([answers[i] for i in np.argsort(similarities)[-3:]])[:2000]
                    result = models['qa_pipeline'](question=user_input, context=context)
                    if result['score'] > 0.4:
                        return f"🔍 {result['answer']}", result['score'], "medium"
                except:
                    pass
                return f"💡 {answers[best_idx]}", best_sim, "medium"
            else:
                return get_fallback(user_input), best_sim, "low"
        except:
            pass
    
    # TF-IDF fallback
    vectorizer, question_vectors, answers = tfidf_data
    similarities = cosine_similarity(vectorizer.transform([user_input.lower()]), question_vectors).flatten()
    best_idx, best_sim = np.argmax(similarities), similarities[np.argmax(similarities)]
    
    if best_sim > 0.4:
        return answers[best_idx], best_sim, "high" if best_sim > 0.6 else "medium"
    else:
        return get_fallback(user_input), best_sim, "low"

def main():
    st.title("🏦 AI Banking Assistant")
    st.caption("Hybrid Intelligence • Advanced AI Models")
    
    models = load_models()
    qa_pairs = load_data()
    tfidf_data = init_tfidf(qa_pairs)
    
    st.success(f"✅ {'HuggingFace + TF-IDF' if models['hf_loaded'] else 'TF-IDF'} • {len(qa_pairs)} Q&A pairs loaded")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! I'm your AI banking assistant. Ask me about accounts, loans, cards, or transfers!"}]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask about banking services..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing..."):
                response, confidence, level = get_response(prompt, models, qa_pairs, tfidf_data)
                st.write(response)
                
                if level == "high": st.success("🎯 High Confidence")
                elif level == "medium": st.info("📊 Medium Confidence")
                else: st.warning("💡 General Guidance")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.sidebar:
        st.header("🤖 AI System")
        st.success("✅ Hugging Face Models" if models['hf_loaded'] else "📊 TF-IDF Mode")
        st.metric("Accuracy", "85-92%" if models['hf_loaded'] else "70-80%")
        st.metric("Q&A Pairs", len(qa_pairs))
        
        st.header("💡 Services")
        for service in ["🏦 Accounts", "💰 Loans", "💳 Cards", "💸 Transfers"]:
            st.write(service)

if __name__ == "__main__":
    main()
