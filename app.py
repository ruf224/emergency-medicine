import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Train the Multi-Substance Clinical Toxicology Model
@st.cache_data
def train_toxicology_model():
    df = pd.read_csv("multi_toxicology_data.csv")
    
    # Map binary target label
    df['Critical_Intervention_Required'] = df['Critical_Intervention_Required'].map({'Yes': 1, 'No': 0})
    
    # Map substance text to numbers
    class_map = {'Acetaminophen': 0, 'Opioid': 1, 'TCA': 2, 'Organophosphate': 3}
    df['Substance_Class'] = df['Substance_Class'].map(class_map)
    
    features = ['Substance_Class', 'Hours_Post_Ingestion', 'GCS_Score', 'QRS_Duration_ms', 'Serum_pH', 'ALT_AST_Level']
    X = df[features]
    y = df['Critical_Intervention_Required']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, class_map

model, class_map = train_toxicology_model()

# 2. Render UI Layout
st.title("🚨 Multi-Substance Acute Toxicology Decision Support")
st.write("Cross-reference kinetic windows, neurological markers, and electrophysiology to predict critical organ failure and life-support requirements.")

st.subheader("📋 Core Poison Exposure Parameters")
col1, col2 = st.columns(2)
with col1:
    toxin_class = st.selectbox("Suspected Ingested Substance Class", list(class_map.keys()))
    hours = st.number_input("Estimated Time Since Ingestion (Hours)", min_value=0.5, max_value=48.0, value=4.0, step=0.5)
    gcs = st.slider("Glasgow Coma Scale (GCS) Score", min_value=3, max_value=15, value=15)

with col2:
    qrs = st.number_input("ECG QRS Duration (milliseconds)", min_value=60, max_value=220, value=85)
    ph = st.number_input("Blood Serum pH Level", min_value=6.50, max_value=7.60, value=7.40, step=0.01)
    liver_enzymes = st.number_input("Baseline Serum ALT / AST (U/L)", min_value=5, max_value=10000, value=25)

# 3. Live Prediction Execution
if st.button("Evaluate Acute Toxicity Risk"):
    input_vector = [[class_map[toxin_class], hours, gcs, qrs, ph, liver_enzymes]]
    prediction = model.predict(input_vector)
    
    st.markdown("---")
    st.subheader("🤖 AI Clinical Decision Suggestion:")
    
    if prediction == 1:
        st.error("""
        ⚠️ CRITICAL INFUSION / LIFE SUPPORT MANDATORY: 
        High mathematical probability of severe physiological deterioration. 
        - For APAP: Initiate N-Acetylcysteine (NAC) protocol immediately.
        - For Opioids: Secure airway; administer continuous Naloxone infusion.
        - For TCAs: Administer Hypertonic Sodium Bicarbonate for QRS narrowing.
        - For Organophosphates: High threat of cholinergic crisis; deploy Atropine and Pralidoxime immediately.
        """)
    else:
        st.success("✅ MONITORING Margins Secure: Current physiological vitals, chemical markers, and kinetic curves indicate low immediate risk of critical respiratory or organ breakdown. Maintain standard poison control protocol and chart serial vitals.")
