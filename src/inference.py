import numpy as np
import pandas as pd
import os
import torch 
import streamlit as st
import models
import dataset
import io
from utils import get_selected_text

def get_test_loader(df):
    loader = torch.utils.data.DataLoader(dataset.TweetDataset(df))
    return loader

def predict(df, model):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    df['text'] = df['text'].astype(str)
    pred_loader = get_test_loader(df)
    predictions = []

    for data in pred_loader:
        ids = data['ids'].to(device)
        masks = data['masks'].to(device)
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
            predictions.append(pred)

    return predictions