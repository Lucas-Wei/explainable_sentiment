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

def predict(df, model):
    df['text'] = df['text'].astype(str)
    pred_loader = get_test_loader(df)
    predictions = []

    for data in test_loader:
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
            predictions.append(pred)

    return predictions

    #data = next(iter(test_loader))