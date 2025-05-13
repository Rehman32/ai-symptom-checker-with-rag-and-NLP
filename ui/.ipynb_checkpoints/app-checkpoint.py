import streamlit as st
import sys
import os

# Add backend folder to path
sys.path.append(os.path.abspath("../backend"))

# Import your parser
from symptom_parser import clean_and_extract_symptoms

st.set_page_config(page_title="AI Symptom Checker", layout="centered")
st.title("🧠 AI-Powered Symptom Checker")

user_input = st.text_area("Enter your symptoms below:", placeholder="e.g. fever, sore throat, body ache")

if st.button("Check Diagnosis"):
    if user_input.strip() == "":
        st.warning("Please enter your symptoms before submitting.")
    else:
        st.info(f"🔍 You entered: {user_input}")
        extracted_symptoms = clean_and_extract_symptoms(user_input)

        if extracted_symptoms:
            st.success(f"✅ Parsed Symptoms: {', '.join(extracted_symptoms)}")
        else:
            st.error("❌ No known symptoms found in your input. Try rephrasing.")
