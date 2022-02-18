from trainer import Trainer
from configs import Config
import torch
from torch import nn, optim
import pandas as pd
from utils import get_augmentation_list, \
                    ImageDataset, \
                    set_seed
from torch.utils.data import DataLoader
import os
from modules.zoo import *
from modules import Model


config = Config()
set_seed(config.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
if device == 'cuda':
    NUM_GPU = torch.cuda.device_count()
    
    if NUM_GPU > 1:
        DISTRIBUTED = True
        config.BATCH_SIZE = config.BATCH_SIZE * NUM_GPU
    else:
        DISTRIBUTED = False


 # === DATA LOADING ===
    
data_csv = pd.read_csv(config.CSV_PATH)

train_transforms = get_augmentation_list(input_size=config.INPUT_DIM)
train_dataset = ImageDataset(data_csv,
                            config.IAMGES_PATH,
                            transform=train_transforms)

train_loader = DataLoader(train_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=os.cpu_count(),
                          pin_memory=True if device == 'cuda' else False)



# ===  MODEL SETUP ===

backbone, backbone_dim = get_backbone(config.BACKBONE_NAME, config.BACKBONE_PARAMS)
pooling = get_pooling(config.POOLING_NAME)
head = get_head(config.HEAD_NAME, config.HEAD_PARAMS)

model = Model(config.CLASS_NUM, backbone, pooling, head, embed_dim=config.CLASS_NUM).to(device)

# === TRAINING SETUP

loss = nn.CrossEntropyLoss()
schedule = get_scheduler(config.SCHEDULER_NAME)

optimizer = optim.Adam(model.parameters(), lr=schedule[0])

if config.IS_RESUME:
    model = torch.load(config.LOAD_PATH).to(device)


# === START TRAINING

trainer = Trainer(criterion=config.loss,
                  optimizer=config.optimizer,
                  device=config.device,
                  start_epoch=config.start_epoch)

trainer.run(config.model, config.train_loader, epochs=config.num_epoch, save_path=config.save_path, schedule=config.schedule)

