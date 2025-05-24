#backend/symptom_checker.py

import re

# Predefined list of known symptoms (you’ll expand this later from dataset)
KNOWN_SYMPTOMS = [
    "fever", "headache", "cough", "sore throat", "fatigue",
    "shortness of breath", "nausea", "vomiting", "diarrhea",
    "body ache", "chills", "loss of appetite", "runny nose",
    "congestion", "rash", "dizziness"
]

def clean_and_extract_symptoms(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation

    extracted = []
    for symptom in KNOWN_SYMPTOMS:
        if symptom in text:
            extracted.append(symptom)
    
    return extracted
