from sklearn.model_selection import StratifiedKFold
import pandas as pd
from model import TweetRobertaModel
import torch
import torch.optim
import configparser
import engine
from dataset import TweetDataset

config = configparser.ConfigParser()
config.read('../config/config.ini')

N_SPLITS = int(config['MODEL']['N_SPLITS'])
BATCH_SIZE = int(config['MODEL']['BATCH_SIZE'])
LR = float(config['MODEL']['LR'])
TRAINING_FILE = config['PATHS']['TRAINING_FILE']
NUM_WORKERS = int(config['MODEL']['NUM_WORKERS'])


def get_train_val_loaders(df, train_idx, val_idx, batch_size=BATCH_SIZE):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS)
    
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    
    return dataloaders_dict

def run():
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)
    train_df = pd.read_csv(TRAINING_FILE)
    train_df = train_df[:100]
    train_df['text'] = train_df['text'].astype(str)
    train_df['selected_text'] = train_df['selected_text'].astype(str)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
        print(f'Fold: {fold}')
        model = TweetRobertaModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
        criterion = engine.loss_fn
        dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, BATCH_SIZE)
        
        engine.train_fn(
            model,
            dataloaders_dict,
            criterion,
            optimizer,
            BATCH_SIZE,
            '../config/roberta-pths/' + f'roberta_fold{fold+1}.pth')

if __name__ == '__main__':
	run()