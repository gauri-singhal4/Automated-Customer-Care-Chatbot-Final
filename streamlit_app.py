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

@st.cache_resource
def load_models():
    if HF_AVAILABLE:
        try:
            return {
                'similarity': SentenceTransformer('all-MiniLM-L6-v2'),
                'qa': pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
                'hf': True
            }
        except:
            pass
    return {'hf': False}

@st.cache_data
def load_data():
    qa_pairs = []
    for path in ["Bank_FAQs.csv", "synthetic_bank_queries_responses(Sheet1).csv", "synthetic_bank_customers(Sheet1).csv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                q_col = next((c for c in df.columns if c.lower() in ['question', 'query']), None)
                a_col = next((c for c in df.columns if c.lower() in ['answer', 'response']), None)
                if q_col and a_col:
                    qa_pairs.extend([{"q": str(row[q_col]), "a": str(row[a_col])} 
                                   for _, row in df.iterrows() if str(row[q_col]) != 'nan'])
            except:
                continue
    
    return qa_pairs or [
        {"q": "How do I open a bank account?", "a": "Visit branch with ID and address proof."},
        {"q": "How to apply for credit card?", "a": "Apply online or visit branch with income proof."},
        {"q": "How do I check my balance?", "a": "Use mobile app, ATM, or call customer service."}
    ]

def get_response(user_input, models, qa_pairs):
    if not user_input.strip():
        return "Please ask me a banking question."
    
    fallbacks = {
        'account': "I can help with account services. What specifically do you need?",
        'loan': "I can assist with loan information. What type of loan interests you?",
        'card': "I can help with card services. What do you need assistance with?",
        'transfer': "I can help with money transfers. What type of transfer do you need?"
    }
    
    if models.get('hf'):
        try:
            questions, answers = [qa['q'] for qa in qa_pairs], [qa['a'] for qa in qa_pairs]
            similarities = np.dot(models['similarity'].encode([user_input]), 
                                models['similarity'].encode(questions).T)[0]
            best_idx, best_sim = np.argmax(similarities), similarities[np.argmax(similarities)]
            
            if best_sim > 0.6:
                return answers[best_idx]
            elif best_sim > 0.4:
                try:
                    context = " ".join([answers[i] for i in np.argsort(similarities)[-3:]])[:2000]
                    result = models['qa'](question=user_input, context=context)
                    if result['score'] > 0.3:
                        return result['answer']
                except:
                    pass
        except:
            pass
    
    # TF-IDF fallback
    questions = [qa['q'].lower() for qa in qa_pairs]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        vectors = vectorizer.fit_transform(questions)
        similarities = cosine_similarity(vectorizer.transform([user_input.lower()]), vectors).flatten()
        best_sim = similarities.max()
        if best_sim > 0.3:
            return [qa['a'] for qa in qa_pairs][similarities.argmax()]
    except:
        pass
    
    # Keyword fallback
    for key, response in fallbacks.items():
        if key in user_input.lower():
            return response
    
    return "I can help with banking services like accounts, loans, cards, and transfers. What do you need help with?"

def main():
    st.title("🏦 AI Banking Assistant")
    
    models = load_models()
    qa_pairs = load_data()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your banking needs today?"}]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask me about banking services..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            response = get_response(prompt, models, qa_pairs)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
