import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"

import sys
print("PYTHON:", sys.executable)

import streamlit as st
import joblib
import pandas as pd

# Add the parent directory to sys.path so Python can find the 'backend' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.rag_helper import get_disease_info  # ✅ works with fixed sys.path

# Load trained model and label encoder from backend folder
model = joblib.load(os.path.join('..', 'backend', 'disease_model.pkl'))
label_encoder = joblib.load(os.path.join('..', 'backend', 'label_encoder.pkl'))

# Load symptoms from Training.csv
df = pd.read_csv(os.path.join('..', 'data', 'Training.csv'))
symptom_columns = df.columns[:-1]  # exclude 'prognosis' column

# App config
st.set_page_config(page_title="AI Symptom Checker", layout="wide")
st.title("🩺 AI-Powered Symptom Checker with Explanations")
st.write("Select your symptoms from the list below. The system will predict the most likely disease and explain it using medical knowledge.")

# UI input
selected_symptoms = st.multiselect("Select Symptoms:", options=list(symptom_columns))

if st.button("Predict Disease"):
    # Encode symptoms into input vector
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]

    # Predict disease
    prediction = model.predict([input_vector])
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🤖 Predicted Disease: **{predicted_label}**")

    # RAG: Generate explanation
    with st.spinner("🔍 Retrieving medical explanation..."):
        disease_info = get_disease_info(predicted_label)

    st.subheader("📖 About This Disease")
    st.write(disease_info)
