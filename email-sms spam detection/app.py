import streamlit as st
import pickle 
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import string
nltk.download('stopwords')

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i) 
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))        
    
    return " ".join(y) 

tfidf= pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms=st.text_area("Enter the message")
if  st.button('Predict'):
    if input_sms:
        transformed_sms=transform_text(input_sms)
        vector_input= tfidf.transform([transformed_sms])
        result=model.predict(vector_input)[0]
        if result == 1:
            st.header("Spam Message!")
        else:
            st.header("Not a Spam Message!")
else:
    st.warning("Please Enter a message!")