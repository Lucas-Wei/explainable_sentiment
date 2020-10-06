from sklearn.model_selection import StratifiedKFold
import pandas as pd
from model import TweetModel
import torch.optim
import configparser
from engine import train_fn

config = configparser.ConfigParser()
config.read('../config/config.ini')


def run()
    skf = StratifiedKFold(n_splits=config[N_SPLITS], shuffle=True, random_state=seed)
    train_df = pd.read_csv(TRAINING_FILE)
    train_df['text'] = train_df['text'].astype(str)
    train_df['selected_text'] = train_df['selected_text'].astype(str)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
        print(f'Fold: {fold}')
        model = TweetModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config[LR], betas=(0.9, 0.999))
        criterion = loss_fn
        dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, int(config[MODEL][BATCH_SIZE])
        
        train_fn(
            model,
            dataloaders_dict,
            criterion,
            optimizer,
            int(config[MODEL][BATCH_SIZE],
            f'roberta_fold{fold}.pth')