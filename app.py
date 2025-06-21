import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
with open('sentiment_model_multiclass.pkl','rb') as f:
    model,vectorizer,label_mapping=pickle.load(f)
inverse_label_mapping={v:k for k,v in label_mapping.items()}
def preprocess_data(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    tokens=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

st.title("Sentiment Analysis")
st.write("Enter a text below to predict sentiment")
user_input=st.text_area("Enter your review :")
if(st.button("Analyze")) :
    if user_input.strip()=="" :
        st.warning("Please enter some text")
    else:
        clean_text=preprocess_data(user_input)
        vectorized_text=vectorizer.transform([clean_text])
        prediction=model.predict(vectorized_text)[0]
        sentiment=inverse_label_mapping[prediction]
        if sentiment == 'Positive':
            st.success(f"Sentiment: {sentiment} 😊")
        elif sentiment == 'Negative':
            st.error(f"Sentiment: {sentiment} 😞")
        else:
            st.info(f"Sentiment: {sentiment} 😐")


