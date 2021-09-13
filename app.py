import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title('Hello World!')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
form = st.form(key='my_form')
text = form.text_input(label='Enter some text')
submit_button = form.form_submit_button(label='Submit')

if submit_button:
    st.subheader('Data')
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    st.write({"emotion": outputs})
