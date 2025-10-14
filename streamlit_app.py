import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    import faiss
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

st.set_page_config(page_title="Banking Assistant", page_icon="🏦")

@st.cache_resource
def load_all_models():
    models = {}
    status = {}
    
    if not MODELS_AVAILABLE:
        return models, {"error": "Models not available"}
    
    # Model 1: Sentence Transformers
    try:
        models['embeddings'] = SentenceTransformer('all-MiniLM-L6-v2')
        status['embeddings'] = "✅"
    except:
        status['embeddings'] = "❌"
    
    # Model 2: DistilBERT Q&A
    try:
        models['qa'] = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        status['qa'] = "✅"
    except:
        status['qa'] = "❌"
    
    # Model 3: DistilGPT-2 Text Generation
    try:
        models['generator'] = pipeline("text-generation", model="distilgpt2", max_length=100, 
                                     num_return_sequences=1, temperature=0.7, pad_token_id=50256)
        status['generator'] = "✅"
    except:
        status['generator'] = "❌"
    
    # Model 4: BART Classifier
    try:
        models['classifier'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        status['classifier'] = "✅"
    except:
        status['classifier'] = "❌"
    
    # Model 5: RoBERTa Sentiment
    try:
        models['sentiment'] = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        status['sentiment'] = "✅"
    except:
        status['sentiment'] = "❌"
    
    return models, status

@st.cache_data
def load_banking_data():
    paths = ["Bank_FAQs.csv", "app/Bank_FAQs.csv"]
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                q_cols = [c for c in df.columns if 'question' in c.lower() or 'query' in c.lower()]
                a_cols = [c for c in df.columns if 'answer' in c.lower() or 'response' in c.lower()]
                if q_cols and a_cols:
                    data = []
                    blacklist = ["customer_", "dear customer", "hdfc", "sbi", "icici", "axis"]
                    for _, row in df.iterrows():
                        q, a = str(row[q_cols[0]]).strip(), str(row[a_cols[0]]).strip()
                        if len(q) > 5 and len(a) > 15 and not any(b in a.lower() for b in blacklist):
                            clean_a = a.replace("HDFC", "your bank").replace("SBI", "your bank").replace("ICICI", "your bank")
                            data.append({"q": q, "a": clean_a})
                    return data
            except: 
                continue
    
    # Enhanced default banking knowledge
    return [
        {"q": "How to open savings account step by step?", "a": "To open a savings account: 1) Choose your bank and account type 2) Gather required documents (ID proof, address proof, photos) 3) Visit branch or apply online 4) Fill application form completely 5) Submit documents for verification 6) Complete KYC process 7) Make initial deposit 8) Receive account number and debit card in 7-10 days."},
        {"q": "What are current home loan interest rates?", "a": "Home loan interest rates currently range from 8.5% to 12% per annum. Rates vary based on your credit score, loan amount, tenure, and bank policies. Fixed rates remain constant, floating rates change with market conditions. Compare rates from multiple banks before deciding."},
        {"q": "How to apply for credit card online?", "a": "Online credit card application: 1) Visit bank's official website 2) Choose suitable card type 3) Check eligibility criteria 4) Fill online application form 5) Upload required documents (ID, income proof, address proof) 6) Submit application 7) Complete verification process 8) Receive approval decision in 7-15 days 9) Card delivered to registered address."},
        {"q": "What are different money transfer methods?", "a": "Money transfer options: 1) NEFT (National Electronic Funds Transfer) - free, takes 2-4 hours 2) RTGS (Real Time Gross Settlement) - instant, for amounts above ₹2 lakhs 3) IMPS (Immediate Payment Service) - instant, available 24/7 4) UPI (Unified Payments Interface) - instant peer-to-peer transfers via mobile apps. All available through internet banking."},
        {"q": "What documents needed for personal loan?", "a": "Personal loan documentation: 1) Identity proof (PAN card, Aadhaar, passport) 2) Address proof (utility bills, rent agreement) 3) Income proof (salary slips for last 3 months, bank statements) 4) Employment certificate or offer letter 5) Recent passport-size photographs 6) Form 16 or ITR for income verification. Additional documents may be required based on loan amount and bank policies."}
    ]

@st.cache_resource
def build_search_system(data, models):
    questions = [item["q"] for item in data]
    answers = [item["a"] for item in data]
    
    # Build TF-IDF (always available)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(questions)
    
    search_system = {
        "vectorizer": vectorizer, "tfidf": tfidf_matrix, 
        "questions": questions, "answers": answers, "data": data
    }
    
    # Add semantic search if embeddings model available
    if 'embeddings' in models:
        try:
            embeddings = models['embeddings'].encode(questions)
            norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            faiss_index = faiss.IndexFlatIP(norm_embeddings.shape[1])
            faiss_index.add(norm_embeddings.astype('float32'))
            search_system.update({
                "embeddings_model": models['embeddings'],
                "faiss_index": faiss_index,
                "embeddings": norm_embeddings
            })
        except:
            pass
    
    return search_system

def generate_multi_model_response(query, search_system, models):
    responses = []
    methods_used = []
    
    # Model 1: Semantic Search (Sentence Transformers + FAISS)
    if "embeddings_model" in search_system:
        try:
            query_emb = search_system["embeddings_model"].encode([query])
            query_emb = query_emb / np.linalg.norm(query_emb)
            distances, indices = search_system["faiss_index"].search(query_emb.astype('float32'), 3)
            if distances[0][0] > 0.7:
                semantic_response = search_system["answers"][indices[0][0]]
                responses.append((semantic_response, distances[0][0], "Semantic Search"))
                methods_used.append("Sentence Transformers")
        except:
            pass
    
    # Model 2: Question Answering (DistilBERT)
    if 'qa' in models:
        try:
            # Create context from top banking answers
            context = " ".join(search_system["answers"][:5])
            qa_result = models['qa'](question=query, context=context)
            if qa_result['score'] > 0.3:
                qa_response = f"{qa_result['answer']}. For more details, please contact your bank directly."
                responses.append((qa_response, qa_result['score'], "Q&A Model"))
                methods_used.append("DistilBERT Q&A")
        except:
            pass
    
    # Model 3: Intent Classification (BART)
    if 'classifier' in models:
        try:
            banking_intents = ["account opening", "loan information", "credit cards", 
                             "money transfers", "document requirements", "interest rates"]
            classification = models['classifier'](query, banking_intents)
            if classification['scores'][0] > 0.8:
                intent = classification['labels'][0]
                intent_responses = {
                    "account opening": "To open a bank account, visit any branch with ID and address proof. Fill the application form and make initial deposit. Account will be activated within 24-48 hours.",
                    "loan information": "Loan interest rates vary by type: Personal loans (10-24%), Home loans (8-12%), Car loans (7-15%). Eligibility depends on credit score, income, and employment stability.",
                    "credit cards": "Credit card application requires age 18+, regular income, and good credit score. Apply online or visit branch with income proof and ID documents.",
                    "money transfers": "Money transfer options include NEFT (free), RTGS (instant), IMPS (24/7), and UPI (mobile). All available through internet banking.",
                    "document requirements": "Common documents: ID proof (PAN, Aadhaar), address proof (utility bills), income proof (salary slips), and recent photographs.",
                    "interest rates": "Interest rates vary by product and bank. Check with multiple banks for current rates as they change frequently based on RBI guidelines."
                }
                intent_response = intent_responses.get(intent, "")
                if intent_response:
                    responses.append((intent_response, classification['scores'][0], "Intent Classification"))
                    methods_used.append("BART Classifier")
        except:
            pass
    
    # Model 4: Enhanced Text Generation (DistilGPT-2)
    if 'generator' in models and responses:
        try:
            # Use best response as seed for generation
            best_base = max(responses, key=lambda x: x[1])[0]
            generation_prompt = f"Banking question: {query}\nAnswer: {best_base[:50]}"
            generated = models['generator'](generation_prompt, max_length=150, do_sample=True)
            generated_text = generated[0]['generated_text']
            
            # Extract only the generated part
            if "Answer:" in generated_text:
                enhanced_response = generated_text.split("Answer:")[-1].strip()
                if len(enhanced_response) > 20 and len(enhanced_response.split()) > 5:
                    responses.append((enhanced_response, 0.8, "Enhanced Generation"))
                    methods_used.append("DistilGPT-2")
        except:
            pass
    
    # Model 5: TF-IDF Search (Enhanced)
    try:
        query_vec = search_system["vectorizer"].transform([query])
        similarities = cosine_similarity(query_vec, search_system["tfidf"]).flatten()
        best_idx = np.argmax(similarities)
        if similarities[best_idx] > 0.3:
            tfidf_response = search_system["answers"][best_idx]
            responses.append((tfidf_response, similarities[best_idx], "TF-IDF Search"))
            methods_used.append("TF-IDF")
    except:
        pass
    
    # Select best response
    if responses:
        best_response = max(responses, key=lambda x: x[1])
        return best_response[0], best_response[2], methods_used
    
    # Ultimate fallback
    fallback_response = generate_fallback(query)
    return fallback_response, "Fallback System", ["Keyword Matching"]

def analyze_sentiment(query, models):
    if 'sentiment' in models:
        try:
            sentiment_result = models['sentiment'](query)
            return sentiment_result[0]['label'], sentiment_result[0]['score']
        except:
            return "NEUTRAL", 0.5
    return "NEUTRAL", 0.5

def generate_fallback(query):
    query_lower = query.lower()
    if "account" in query_lower or "open" in query_lower:
        return "To open a bank account, visit any branch with valid ID and address proof. Fill the application form and make initial deposit. Account activation takes 1-2 days."
    elif "loan" in query_lower or "rate" in query_lower:
        return "Loan interest rates vary by type and bank. Personal loans: 10-24%, Home loans: 8-12%. Check with multiple banks for current rates and eligibility criteria."
    elif "card" in query_lower or "credit" in query_lower:
        return "Credit card application requires age 18+, regular income, and good credit score. Apply online or visit bank branch with required documents."
    elif "transfer" in query_lower or "payment" in query_lower:
        return "Money transfer options: NEFT (free), RTGS (instant), IMPS (24/7), UPI (mobile). All available through internet banking and mobile apps."
    else:
        return "I can help with banking services including accounts, loans, credit cards, and money transfers. Please ask about specific banking needs."

def main():
    st.title("🏦 Banking Assistant")
    
    # Load all models and data
    with st.spinner("Loading 5 AI models..."):
        models, model_status = load_all_models()
        banking_data = load_banking_data()
        search_system = build_search_system(banking_data, models)
    
    # Initialize chat
    if "messages" not in st.session_state:
        active_models = len([v for v in model_status.values() if v == "✅"])
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hi! I'm powered by {active_models} AI models. Ask me about banking services like accounts, loans, cards, or transfers."}
        ]
    
    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about banking..."):
        # Analyze sentiment
        sentiment_label, sentiment_score = analyze_sentiment(prompt, models)
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response using all models
        with st.chat_message("assistant"):
            with st.spinner("Processing with AI models..."):
                response, method, models_used = generate_multi_model_response(prompt, search_system, models)
            
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
