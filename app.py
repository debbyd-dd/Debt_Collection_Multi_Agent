import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Debt Recovery Agent", page_icon="🤖", layout="wide")

# --- CACHING MODELS (So the app runs fast for the client) ---
@st.cache_resource
def load_nlp_model():
    """Loads the Deep Learning NLP model once and caches it."""
    # Using a smaller, faster model for web deployment demo
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def train_ml_model():
    """Simulates data and trains the ML model once."""
    np.random.seed(42)
    n_records = 1000
    data = pd.DataFrame({
        'debt_amount': np.random.uniform(500, 10000, n_records),
        'days_past_due': np.random.randint(30, 365, n_records),
        'num_previous_contacts': np.random.randint(1, 15, n_records),
        'recent_payment_history': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),
    })
    
    prob_settle = (data['recent_payment_history'] * 0.4) + (1000 / data['debt_amount']) + (30 / data['days_past_due'])
    prob_settle = np.clip(prob_settle, 0.1, 0.9)
    data['settled_debt'] = np.random.binomial(1, prob_settle)
    
    X = data.drop('settled_debt', axis=1)
    y = data['settled_debt']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# Load models
nlp_agent = load_nlp_model()
ptp_model, scaler = train_ml_model()

collection_intents = ["willing to pay", "financial hardship", "disputing debt", "refusal to pay", "requesting more info"]

# --- WEB UI FRONTEND ---
st.title("🤖 Multi-Model AI Agent: Debt Recovery Demo")
st.markdown("""
This dashboard demonstrates a multi-agent AI approach. 
* **Model 1 (Machine Learning):** Predicts Propensity to Pay (PTP) based on financial metrics.
* **Model 2 (Deep Learning):** Analyzes unstructured text to determine the debtor's intent.
* **Decision Engine:** Combines both outputs to recommend an automated action.
""")

st.divider()

# Create two columns for the layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Debtor Financial Profile")
    st.markdown("Adjust these sliders to simulate different debtor profiles.")
    
    debt_amount = st.number_input("Debt Amount ($)", min_value=100, max_value=20000, value=8500, step=100)
    days_past_due = st.slider("Days Past Due", min_value=1, max_value=365, value=120)
    num_contacts = st.slider("Previous Contacts Attempted", min_value=0, max_value=20, value=8)
    recent_payment = st.selectbox("Has made a payment in last 90 days?", options=["No", "Yes"])
    
    recent_payment_val = 1 if recent_payment == "Yes" else 0
    debtor_profile = [debt_amount, days_past_due, num_contacts, recent_payment_val]

with col2:
    st.subheader("✉️ Debtor Communication")
    st.markdown("Paste a simulated email, SMS, or call transcript from the debtor.")
    
    default_text = "I do not owe this money. The service was never provided. Stop calling me."
    communication = st.text_area("Debtor's Message", value=default_text, height=150)
    
    analyze_button = st.button("🧠 Analyze Account & Get Recommendation", type="primary", use_container_width=True)

# --- AI ANALYSIS LOGIC ---
if analyze_button:
    with st.spinner("Multi-Agent AI is analyzing the account..."):
        
        # 1. Run ML Model
        profile_scaled = scaler.transform([debtor_profile])
        ptp_prob = ptp_model.predict_proba(profile_scaled)[0][1] * 100
        
        # 2. Run NLP Model
        intent_result = nlp_agent(communication, collection_intents)
        top_intent = intent_result['labels'][0]
        intent_confidence = intent_result['scores'][0] * 100
        
        st.divider()
        st.subheader("📈 AI Analysis Results")
        
        # Display Metrics in a row
        m1, m2 = st.columns(2)
        m1.metric("Propensity to Pay (ML Model)", f"{ptp_prob:.1f}%")
        m2.metric("Detected Intent (NLP Model)", top_intent.title(), f"{intent_confidence:.1f}% confidence")
        
        # 3. Decision Engine Logic
        st.subheader("🎯 Agent Recommendation")
        
        if top_intent == "disputing debt":
            st.error("**ACTION:** HALT automated collections. Route immediately to Legal/Dispute Resolution team.")
        elif top_intent == "financial hardship" and ptp_prob > 40:
            st.warning("**ACTION:** Send 'Hardship Settlement Plan' via email (Offer 6-month payment plan).")
        elif top_intent == "willing to pay":
            st.success("**ACTION:** Send automated SMS with direct secure payment link.")
        elif ptp_prob < 20 and top_intent == "refusal to pay":
            st.error("**ACTION:** Account deemed unrecoverable via standard means. Flag for credit bureau reporting or debt sale.")
        else:
            st.info("**ACTION:** Queue for standard human agent follow-up call tomorrow at 10 AM.")
