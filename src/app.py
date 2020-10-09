import numpy as np
import pandas as pd
import base64
import os
import torch 
import streamlit as st
import models
import configparser
import dataset
import io
import inference
import configparser

config = configparser.ConfigParser()
config.read('../config/config.ini')

PTHS_PATH = config['PATHS']['PTHS_PATH']

st.title('Explainable Sentiment')
st.sidebar.title('App Mode')
selection = st.sidebar.radio("Option:", ['Single Text & Sentiment', 'Upload File (.csv)'])

@st.cache
def load_model():
    model = models.TweetRoBERTaModel()
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(PTHS_PATH, 'RoBERTa_fold1.pth')))
    model.eval()
    return model

model = load_model()

def single_mode():
	'### Single Text & Sentiment'
	'#### Text here:'
	input_text = st.text_area('', max_chars=144)

	'#### Sentiment:'
	sentiment = st.selectbox('',('negative', 'neutral', 'positive'))

	if input_text:
		d = {'text': input_text, 'sentiment': sentiment}
		df = pd.DataFrame(data=d, index=[0])
		pred = inference.predict(df, model=model)

		'### Words explain sentiment:'
		st.write(pred[0])

def file_mode():
	st.set_option('deprecation.showfileUploaderEncoding', False)

	uploaded_file = st.file_uploader("", type='csv')
	if uploaded_file:
		df = pd.read_csv(uploaded_file, usecols=['text', 'sentiment'])
		st.dataframe(df.head(5)) # show first 5 rows
		st.write(df.shape[0])

		pred = inference.predict(df, model=model)
		df['support words'] = pred
		st.dataframe(df.head(5)) # show first 5 predictions

		csv = df.to_csv(index=False)

		# create download link
		b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
		href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>' # (right-click and save as &lt;some_name&gt;.csv)
		st.markdown(href, unsafe_allow_html=True)

if selection == 'Single Text & Sentiment':
	single_mode()
elif selection == 'Upload File (.csv)':
	file_mode()