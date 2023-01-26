import pandas as pd
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

#load save model
model_fraud = pickle.load(open('model_fraud.sav','rb'))

tfidf = TfidfVectorizer                               

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf_idf.sav", "rb"))))

#judul
st.title ('-- Prediksi Komentar Pada Aplikasi Instagram --')

clean_teks = st.text_input('Masukan komentar Instagram')

fraud_detection = ''

if st.button('Hasil Prediksi'):
    predict_fraud = model_fraud.predict(loaded_vec.fit_transform([clean_teks]))

    if (predict_fraud[0] == 0):
        fraud_detection = "Komentar Bersifat Negatif"
    else :
        fraud_detection = "Komentar Bersifat Positif"

st.success(fraud_detection)  

