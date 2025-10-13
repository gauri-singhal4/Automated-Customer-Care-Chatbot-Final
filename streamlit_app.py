import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Try importing with fallback
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Advanced features unavailable: {e}")
    EMBEDDINGS_AVAILABLE = False

st.set_page_config(page_title="🏦 AI Banking Assistant", page_icon="🏦", layout="wide")

@st.cache_data
def load_data():
    paths = ["Bank_FAQs.csv", os.path.join(".", "Bank_FAQs.csv"), os.path.join("app", "Bank_FAQs.csv")]
    df = None
    for path in paths:
        if os.path.exists(path):
            try: df = pd.read_csv(path); break
            except: continue
    
    if df is None:
        return get_enhanced_defaults()
    
    q_cols = [c for c in df.columns if any(x in c.lower() for x in ['question', 'query', 'q'])]
    a_cols = [c for c in df.columns if any(x in c.lower() for x in ['answer', 'response', 'a'])]
    
    if not q_cols or not a_cols:
        return get_enhanced_defaults()
    
    df = df[df[q_cols[0]].notna() & df[a_cols[0]].notna()]
    blacklist = ["customer_", "dear customer", "rs.", "$", "icici", "hdfc", "axis", "sbi"]
    
    qa_pairs = []
    for _, row in df.iterrows():
        q, a = str(row[q_cols[0]]).strip(), str(row[a_cols[0]]).strip()
        if len(q) > 5 and len(a) > 15 and not any(bad in a.lower() for bad in blacklist):
            qa_pairs.append({"q": q, "a": make_generic(a), "cat": categorize(q), "pri": calc_priority(q, a)})
    
    qa_pairs.extend(get_enhanced_defaults())
    return sorted(qa_pairs, key=lambda x: x.get('pri', 5), reverse=True)

def make_generic(answer):
    reps = {"HDFC Bank": "your bank", "HDFC": "your bank", "ICICI Bank": "your bank", "SBI": "your bank", 
            "NetBanking": "internet banking", "our branch": "any bank branch", "we offer": "banks offer"}
    for old, new in reps.items():
        answer = answer.replace(old, new)
    return answer.strip() + ("" if answer.endswith(('.', '!', '?')) else ".")

def categorize(question):
    q = question.lower()
    if "card" in q: return "cards"
    if any(x in q for x in ["loan", "emi", "credit"]): return "loans"
    if any(x in q for x in ["account", "balance", "saving"]): return "accounts"
    if any(x in q for x in ["transfer", "payment", "neft", "rtgs", "upi"]): return "transfers"
    return "general"

def calc_priority(q, a):
    pri = 5 + (2 if len(a.split()) > 20 else 0) + (1 if any(x in a for x in ['1)', 'steps']) else 0)
    return pri + (1 if any(term in q.lower() for term in ['account', 'loan', 'card']) else 0)

def get_enhanced_defaults():
    return [
        {"q": "How to open savings account?", "a": "Visit bank with ID and address proof. Fill form, make initial deposit, complete KYC. Account active in 24-48 hours. Online applications also available.", "cat": "accounts", "pri": 9},
        {"q": "Home loan interest rates?", "a": "Home loan rates: 8.5-12% annually. Depends on credit score, loan amount, tenure. Fixed or floating rates available. Check multiple banks for best rates.", "cat": "loans", "pri": 9},
        {"q": "Credit card application process?", "a": "Credit card steps: Check eligibility → Choose card type → Submit documents → Verification → Approval in 7-15 days → Card delivery. Age 18+, regular income required.", "cat": "cards", "pri": 9},
        {"q": "Online money transfer methods?", "a": "Transfer options: NEFT (2-4 hours, free), RTGS (instant, ₹2L+), IMPS (instant, 24/7), UPI (instant P2P). All via internet/mobile banking.", "cat": "transfers", "pri": 8},
        {"q": "Personal loan documents needed?", "a": "Personal loan docs: ID proof, address proof, income proof (3 months salary slips), bank statements, employment certificate, photos, Form 16/ITR.", "cat": "loans", "pri": 8}
    ]

@st.cache_resource
def build_indexes(qa_pairs):
    questions = [item["q"] for item in qa_pairs]
    answers = [item["a"] for item in qa_pairs]
    priorities = [item.get("pri", 5) for item in qa_pairs]
    
    # Always build TF-IDF (no dependency issues)
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=2000)
    tfidf_mat = vec.fit_transform([q.lower() for q in questions])
    
    result = {"vec": vec, "tfidf": tfidf_mat, "questions": questions, "answers": answers, "priorities": priorities}
    
    # Try to add embeddings if available
    if EMBEDDINGS_AVAILABLE:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(questions, show_progress_bar=False)
            norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            faiss_idx = faiss.IndexFlatIP(norm_emb.shape[1])
            faiss_idx.add(norm_emb.astype('float32'))
            
            result.update({"model": model, "faiss": faiss_idx, "emb": norm_emb, "embeddings_active": True})
        except Exception as e:
            st.warning(f"Embeddings disabled: {e}")
            result["embeddings_active"] = False
    else:
        result["embeddings_active"] = False
    
    return result

def enhanced_search(query, idx):
    # Method 1: Try embeddings if available
    if idx.get("embeddings_active") and "model" in idx:
        try:
            q_emb = idx["model"].encode([query])
            q_emb = q_emb / np.linalg.norm(q_emb)
            emb_dist, emb_idx = idx["faiss"].search(q_emb.astype('float32'), 5)
            
            # TF-IDF search
            q_vec = idx["vec"].transform([query.lower()])
            tfidf_sim = cosine_similarity(q_vec, idx["tfidf"]).flatten()
            
            # Combined scoring
            scores = []
            for i in range(len(idx["answers"])):
                emb_score = emb_dist[0][list(emb_idx[0]).index(i)] if i in emb_idx[0] else 0
                tfidf_score = tfidf_sim[i]
                pri_weight = idx["priorities"][i] / 10.0
                combined = emb_score * 0.6 + tfidf_score * 0.3 + pri_weight * 0.1
                scores.append((combined, i))
            
            best_score, best_idx = max(scores)
            if best_score > 0.4:
                return idx["answers"][best_idx], best_score, "🎯 Hybrid Search"
        except:
            pass
    
    # Method 2: TF-IDF only (fallback)
    q_vec = idx["vec"].transform([query.lower()])
    tfidf_sim = cosine_similarity(q_vec, idx["tfidf"]).flatten()
    
    # Add priority weighting
    weighted_scores = []
    for i, score in enumerate(tfidf_sim):
        weight = idx["priorities"][i] / 10.0
        weighted_scores.append(score + weight * 0.2)
    
    best_idx = np.argmax(weighted_scores)
    best_score = weighted_scores[best_idx]
    
    if best_score > 0.3:
        return idx["answers"][best_idx], best_score, "📊 TF-IDF Search"
    
    return "", 0, "❌ No Match"

def generate_template_response(query):
    templates = {
        "account": ["Account services: Visit branch with documents, fill forms, make deposit. Online options available.", 
                   "To open account: ID proof + address proof + initial deposit. Processing: 1-2 days."],
        "loan": ["Loan info: Rates vary 8-24% based on type and credit. Documents: ID, income, address proof required.",
                "Loan process: Application → Documentation → Verification → Approval → Disbursal. Timeline: 7-15 days."],
        "card": ["Card application: Age 18+, income proof needed. Process: Apply → Verify → Approve → Delivery in 10-15 days.",
                "Cards available: Credit, debit, prepaid. Benefits vary by type. Compare features before applying."],
        "transfer": ["Money transfer: NEFT/RTGS/IMPS/UPI available. NEFT free, others minimal charges. Instant to few hours.",
                    "Transfer methods: Online banking, mobile apps, branch visits. Registration required for beneficiaries."]
    }
    
    q_lower = query.lower()
    for key, responses in templates.items():
        if key in q_lower:
            return random.choice(responses)
    return None

def smart_fallback(query):
    fallbacks = {
        "rate": "Rates vary by product and bank. Check current rates with your preferred bank or RBI website.",
        "fee": "Fees differ by bank and service. Check fee schedule with your bank for accurate information.",
        "eligibility": "Eligibility varies by product. Generally: age, income, credit score matter. Contact bank for details.",
        "document": "Common docs: ID proof, address proof, income proof. Specific needs vary by service.",
        "time": "Timeline varies: Account opening (1-2 days), loans (7-15 days), cards (10-15 days)."
    }
    
    for key, fallback in fallbacks.items():
        if key in query.lower():
            return fallback
    
    return "I can help with banking queries. Please specify: accounts, loans, cards, or transfers for detailed assistance."

def generate_response(query, idx):
    # Method 1: Enhanced retrieval
    result, confidence, method = enhanced_search(query, idx)
    if confidence > 0.5:
        enhanced = result + "\n\nFor specific details, contact your bank directly."
        return enhanced, method
    
    # Method 2: Template responses
    template_resp = generate_template_response(query)
    if template_resp:
        return template_resp, "📝 Template Response"
    
    # Method 3: Smart fallback
    fallback_resp = smart_fallback(query)
    return fallback_resp, "💡 Smart Guidance"

def main():
    st.title("🏦 Enhanced AI Banking Assistant")
    st.caption("*Advanced retrieval with compatibility-focused design*")
    
    # Load and build indexes
    with st.spinner("🔄 Loading banking intelligence..."):
        qa_data = load_data()
        indexes = build_indexes(qa_data)
    
    # Status display
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Knowledge Base", f"{len(qa_data)} Responses")
    with col2: 
        embedding_status = "✅ Active" if indexes.get("embeddings_active") else "📊 TF-IDF Only"
        st.metric("Search Engine", embedding_status)
    with col3: st.metric("Response Quality", "Enhanced")
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", 
            "content": "Hello! I'm your enhanced banking assistant. Ask me about accounts, loans, cards, transfers, or any banking service!"}]
    
    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "method" in msg: st.caption(msg["method"])
    
    # Handle input
    if prompt := st.chat_input("Ask about banking services..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Generating response..."): 
                response, method = generate_response(prompt, indexes)
            st.write(response)
            st.caption(method)
            st.session_state.messages.append({"role": "assistant", "content": response, "method": method})
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Features")
        if indexes.get("embeddings_active"):
            st.write("✅ **Semantic Embeddings**")
            st.write("✅ **FAISS Vector Search**")
        else:
            st.write("📊 **TF-IDF Search Active**")
            st.write("ℹ️ **Embeddings Disabled**")
        st.write("✅ **Multi-Template System**") 
        st.write("✅ **Priority Matching**")
        st.write("✅ **Smart Fallbacks**")
        st.success("🔥 All core features active!")

if __name__ == "__main__":
    main()
