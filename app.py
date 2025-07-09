import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
with open('sentiment_model_multiclass.pkl','rb') as f:
    model,vectorizer,label_mapping=pickle.load(f)
inverse_label_mapping={v:k for k,v in label_mapping.items()}
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")


st.markdown("""
    <style>
        body {
            background-color: #d676ae;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
        }
        h1 {
            color:#9709e3;
            text-align: center;
        }
        .stButton>button {
            background-color:#9709e3;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063827.png", width=100)
st.sidebar.title("🧠 About")
st.sidebar.info("This app uses a Machine Learning model to predict the sentiments of the movie review given!")

st.title("🎯 Sentiment Analyzer -- Real Time Review Analysis 🚀")
st.subheader("🔎 Type your review below...")

user_input = st.text_area("✍️ Enter your review here:")

def preprocess_data(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    tokens=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)
if st.button("Analyze Sentiment 🔍"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(1) 
            clean_text = preprocess_data(user_input)
            vectorized_text = vectorizer.transform([clean_text])
            prediction = model.predict(vectorized_text)[0]
            sentiment = inverse_label_mapping[prediction]
        if sentiment == 'Positive':
            st.success(f"Sentiment: {sentiment} 🎉")
        elif sentiment == 'Negative':
            st.error(f"Sentiment: {sentiment} 😢")
        else:
            st.info(f"Sentiment: {sentiment} 😐")

st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)


