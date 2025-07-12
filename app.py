import streamlit as st
import spacy
import subprocess

@st.cache_resource
def load_model():
    try:
        # Try loading the model
        return spacy.load("en_core_web_md")
    except OSError:
        # If model is not found, download it
        with st.spinner("Downloading spaCy model..."):
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
        return spacy.load("en_core_web_md")

# Load model
nlp = load_model()

# Example usage
st.title("ML Chatbot 🤖")
st.write("This is a demo app using spaCy model `en_core_web_md`.")

# Input
user_input = st.text_input("Ask something:")
if user_input:
    doc = nlp(user_input)
    st.write("Tokens:", [token.text for token in doc])
