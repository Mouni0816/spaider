import streamlit as st
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Sentiment analysis pipeline


# Question answering pipeline, specifying the checkpoint identifier
model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', tokenizer='distilbert-base-uncased-finetuned-sst-2-english')

# Named entity recognition pipeline, passing in a specific model and tokenizer
st.title('Hello World!')


form = st.form(key='my_form')
text = form.text_input(label='Enter some text')
submit_button = form.form_submit_button(label='Submit')

if submit_button:
    st.subheader('Data')
    outputs = model(text)
    st.write({"emotion": outputs})
