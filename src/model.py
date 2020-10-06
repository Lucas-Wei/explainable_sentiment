import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig
import os
import configparser

config = configparser.ConfigParser()
config.read('../config/config.ini')
config_path = config['PATHS']['ROBERTA_PATH']
model_path = config['PATHS']['ROBERTA_PATH']

class TweetRobertaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        roberta_config = RobertaConfig.from_pretrained(
            os.path.join(config_path, 'config.json'),
            output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(
            os.path.join(model_path, 'pytorch_model.bin'), config=roberta_config)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(roberta_config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)
        
    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask)

        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits

if __name__ == "__main__":
    model = TweetRobertaModel()