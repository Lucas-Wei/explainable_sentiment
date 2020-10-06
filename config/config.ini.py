[FILES]
TRAINING_FILE = '../data/train.csv'
ROBERTA_PATH = '../config/roberta-base'
PTHS_PATH = '../config/roberta-pths'

[MODEL]
NUM_EPOCHS = 3
N_SPLITS = 10
MAXLEN = 96
DROP_RATE = 0.5
BATCH_SIZE = 32
NUM_WORKERS = 2
LR = 3e-5