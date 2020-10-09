import numpy as np
import pandas as pd
import os
import torch 
import streamlit as st
import models
import configparser
import dataset
import io

config = configparser.ConfigParser()
config.read('../config/config.ini')

PTHS_PATH = config['PATHS']['PTHS_PATH']

def get_selected_text(text, start_idx, end_idx, offsets):
    if start_pred > end_pred:
        selected_text = tweet
    else:
        selected_text = ""
        for ix in range(start_idx, end_idx + 1):
            selected_text += text[offsets[ix][0]: offsets[ix][1]]
            if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                selected_text += " "
    return selected_text

def get_test_loader(df):
    loader = torch.utils.data.DataLoader(dataset.TweetDataset(df))
    return loader

st.title('Explainable Sentiment')

@st.cache
def build_model():
    model = models.TweetRoBERTaModel()
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(PTHS_PATH, 'RoBERTa_fold1.pth')))
    model.eval()
    return model

model = build_model()

input_text = st.text_area('Text Here', max_chars=144)
sentiment = st.selectbox(
    'Choose Sentiment:',
    ('negative', 'neutral', 'positive'))

if input_text:
    d = {'text': input_text, 'sentiment': sentiment}
    df = pd.DataFrame(data=d, index=[0])
    df['text'] = df['text'].astype(str)
    test_loader = get_test_loader(df)
    data = next(iter(test_loader))

    ids = data['ids'].cuda()
    masks = data['masks'].cuda()
    tweet = data['tweet']
    offsets = data['offsets'].numpy()

    with torch.no_grad():
        output = model(ids, masks)
        start_logits = torch.softmax(output[0], dim=1).cpu().detach().numpy()
        end_logits = torch.softmax(output[1], dim=1).cpu().detach().numpy()
    for i in range(len(ids)):
        start_pred = np.argmax(start_logits)
        end_pred = np.argmax(end_logits)
        pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])

    st.text('Words explain sentiment:')
    st.write(pred)

uploaded_file = st.file_uploader("Choose a file(.csv)", type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # text_io = io.TextIOWrapper(uploaded_file)
    st.dataframe(data.head(10))