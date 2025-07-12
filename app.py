import streamlit as st
import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load SpaCy model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_md")

nlp = load_model()

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("mal_faq.csv")
    df['question'] = df['question'].apply(lambda x: x.lower())
    df['vector'] = df['question'].apply(lambda x: nlp(x).vector)
    return df

df = load_data()

# Chatbot response function
def chatbot_response(user_input):
    input_vector = nlp(user_input.lower()).vector
    similarities = [cosine_similarity([input_vector], [vec])[0][0] for vec in df['vector']]
    max_index = np.argmax(similarities)

    if similarities[max_index] > 0.75:
        return df['answer'][max_index]
    else:
        return "I'm not sure how to respond to that. Try rephrasing."

# Streamlit UI
st.set_page_config(page_title="🧠 ML Chatbot", page_icon="🤖")
st.title("🤖 Machine Learning Chatbot")

user_input = st.text_input("Ask me anything about Machine Learning:")

if user_input:
    response = chatbot_response(user_input)
    st.markdown(f"**Bot:** {response}")
