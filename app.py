import streamlit as st
import sys
import os
import json

# Add project root to path for local imports
sys.path.append(os.getcwd())

from services.ml_predictor import MLPredictor
from services.neural_engine import ClinicalNeuralEngine

st.set_page_config(
    page_title="AI Clinical Decision Support",
    page_icon="🏥",
    layout="wide"
)

# Shared Styling
st.markdown("""
<style>
    .report-card {
        padding: 25px;
        border-radius: 12px;
        background-color: #1a1c23;
        border: 1px solid #2d3139;
        margin-bottom: 20px;
    }
    .ml-tag {
        background: #1e3a5f;
        color: #6fb1fc;
        padding: 4px 10px;
        border-radius: 5px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .neural-tag {
        background: #1e5f3a;
        color: #6ffc9e;
        padding: 4px 10px;
        border-radius: 5px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3a8a, #1e40af);
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1e40af, #1d4ed8);
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def initialize_models():
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = MLPredictor()
    if 'neural_engine' not in st.session_state:
        st.session_state.neural_engine = ClinicalNeuralEngine()

def main():
    initialize_models()
    
    st.title("🏥 Clinical Intelligence System")
    st.caption("Patient-Aware Diagnostic Review & Medication Protocol Engine")

    with st.sidebar:
        st.header("Patient Profile")
        age = st.number_input("Age", 1, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        allergies = st.multiselect("Known Allergies", 
                                  ["Penicillin", "Sulfa Drugs", "Aspirin", "NSAIDs", "Latex", "Codeine"],
                                  default=[])
        
        st.divider()
        st.markdown("### Model Architecture")
        st.info("""
        **V1: NLP Preprocessing**
        - Tokenization & Lemmatization
        
        **V2: ML Classification**
        - Logistic Regression Model
        
        **V3: Neural Synthesis**
        - Large Parameter Transformer
        """)

    # Main Input
    st.subheader("Symptom Input Analysis")
    symptoms = st.text_area("Describe symptoms in medical or plain language:", 
                          placeholder="e.g., Persistent dry cough, chest tightness, shortness of breath for 3 days.",
                          height=100)

    if st.button("RUN CLINICAL INFERENCE"):
        if not symptoms:
            st.warning("Input required for analysis.")
            return

        with st.spinner("Executing Multi-Layer Analysis..."):
            # PHASE 1: ML Classification
            pred_condition, confidence = st.session_state.ml_model.predict_condition(symptoms)
            
            # PHASE 2: Neural Refinement
            analysis = st.session_state.ml_model.predict_condition(symptoms) # Dummy just to show ML working
            
            # Actual LLM call (Neural Engine)
            neural_data = st.session_state.neural_engine.analyze_case(
                symptoms, pred_condition, age, gender, allergies
            )

            # --- Layout Results ---
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div class="report-card">
                    <span class="ml-tag">PHASE 1: ML DIAGNOSTICS</span>
                    <h2 style='margin-top:10px;'>{pred_condition}</h2>
                    <p style='color:#888;'>Model Confidence: {confidence*100:.1f}%</p>
                    <hr style='border-color:#333;'>
                    <span class="neural-tag">PHASE 2: NEURAL REFINEMENT</span>
                    <h3 style='margin-top:10px;'>{neural_data.get('refined_condition')}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if neural_data.get('safety_alerts') != "NONE":
                    st.error(f"**Safety Alert:** \n{neural_data.get('safety_alerts')}")

            with col2:
                st.markdown("### Recommended Clinical Protocols")
                for med in neural_data.get('medications', []):
                    with st.expander(f"💊 {med.get('name')} | {med.get('dose')}", expanded=True):
                        st.write(f"**Rationale:** {med.get('rationale')}")
                        st.caption(f"**Potential Side Effects:** {med.get('side_effects')}")

            st.divider()
            
            with st.container():
                st.markdown("### Clinical Reasoning Summary")
                st.info(neural_data.get('clinical_summary'))
                
            st.divider()
            st.caption(f"⚠️ {neural_data.get('disclaimer')}")

if __name__ == "__main__":
    main()
