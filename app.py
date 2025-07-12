import streamlit as st
import spacy
from spacy.cli import download

@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        # Download properly using spacy's internal CLI
        with st.spinner("Downloading spaCy model..."):
            download("en_core_web_md")
        return spacy.load("en_core_web_md")

# Load model
nlp = load_model()

st.title("ML Chatbot 🤖")
st.write("This is a demo app using spaCy model `en_core_web_md`.")

# Input
user_input = st.text_input("Ask something:")
if user_input:
    doc = nlp(user_input)
    st.write("Tokens:", [token.text for token in doc])
