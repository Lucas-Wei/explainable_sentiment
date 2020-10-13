import torch
from torch import nn
import utils
from tqdm import tqdm
import numpy as np

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss

def train_fn(model, selected_model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            epoch_loss = 0.0
            epoch_jaccard = 0.0
            
            tk0 = tqdm(dataloaders_dict[phase], total=len(dataloaders_dict[phase]))
            for data in (tk0):
                ids = data['ids'].to(device)
                masks = data['masks'].to(device)
                tweet = data['tweet']
                offsets = data['offsets'].numpy()
                start_idx = data['start_idx'].to(device)
                end_idx = data['end_idx'].to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if selected_model == 'LSTM':
                        start_logits, end_logits = model(ids)
                    elif selected_model == 'RoBERTa':
                        start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * len(ids)
                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
                    
                    for i in range(len(ids)):
                        start_pred = np.argmax(start_logits[i])
                        end_pred = np.argmax(end_logits[i])
                        pred = utils.get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                        true = utils.get_selected_text(tweet[i], start_idx[i], end_idx[i], offsets[i])
                        jaccard_score = utils.jaccard(pred, true)
                        epoch_jaccard += jaccard_score
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)
        
            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))
    torch.save(model.state_dict(), filename)

