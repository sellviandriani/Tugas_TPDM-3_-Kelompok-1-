import streamlit as st
import joblib

model_file = open('Instagram Classifier.joblib', 'rb')
model = joblib.load(model_file)

st.write("""
# -- Memprediksi Ulasan pada Aplikasi Instagram --
""")

inputs = st.text_input('Masukkan Teks Komentar')
cek = st.button('Cek Prediksi Komentar')

input_text= [inputs]
def preProcessText(instagram):
    new_ig = []
    for tw in tweer:
        tw = case_folding(tw)
        tw = tokenized(tw)
        tw = stemming(tw)
        tw = removeSlang(tw)
        tw = removeStopWords(tw)
        tw = ' '.join(tw)
        new_ig.append(tw)

    return tw

def predictNewData(ig):
    saved_model = joblib.load('Instagram Classifier.joblib') 
    saved_tfidf = joblib.load('Instagram TF-IDF Vectorizer.joblib') #???
   
    vectorized_ig = saved_tfidf.transform(input_text)
    print(vectorized_ig.shape)
    input_prediction = saved_model.predict(vectorized_ig)
    
    for i in range(len(inputs)):
        if cek:
            if input_prediction[i]:
                st.write('\nPrediksi Ulasan :',input_prediction[i])
            else:
               st.write('\nPrediksi Ulasan :',input_prediction[i])

predictNewData(inputs)
