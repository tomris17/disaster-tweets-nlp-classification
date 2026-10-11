import streamlit as st
import pandas as pd
import re
import string
import joblib

st.set_page_config(
    page_title="Disaster Tweets Classifier",
    page_icon="🚨",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load('disaster_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("🚨 Real or Not? Disaster Tweets Classification")
st.write("Bu uygulama, girilen bir tweet metninin gerçek bir afet durumunu mu yoksa mecazi/yanıltıcı bir ifadeyi mi içerdiğini tahmin eder.")

user_input = st.text_area("Bir Tweet Metni Girin:", "Forest fire near my house, emergency services are on the way!")

if st.button("Sınıflandır"):
    if user_input.strip() == "":
        st.warning("Lütfen geçerli bir metin girin.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        
        st.markdown("---")
        st.subheader("📊 Tahmin Sonucu:")
        if prediction == 1:
            st.error(f"🚨 **GERÇEK AFET / ACİL DURUM** (Güven Skoru: %{probability[1]*100:.2f})")
        else:
            st.success(f"✅ **AFET DEĞİL / MECAZİ** (Güven Skoru: %{probability[0]*100:.2f})")