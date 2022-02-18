from utils.trainer import Trainer
from configs import config
import torch
from torch import nn, optim
import pandas as pd
import os
import json
from utils import get_augmentation_list, \
                    ImageDataset, DummyDataset, \
                    set_seed
from torch.utils.data import DataLoader

from modules.zoo import get_backbone,\
                        get_pooling, \
                        get_head, \
                        get_scheduler

from modules import Model

set_seed(config.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    NUM_GPU = torch.cuda.device_count()
    print(f'[INFO] number of GPUs found: {NUM_GPU}')
    if NUM_GPU > 1:
        DISTRIBUTED = True
        config.BATCH_SIZE = config.BATCH_SIZE * NUM_GPU
    else:
        DISTRIBUTED = False

if not os.path.exists(config.SAVE_PATH):
    os.mkdir(config.SAVE_PATH)

with open(os.path.join(config.SAVE_PATH + '/config.json'), 'w') as f:
    model_config = {x:dict(config.__dict__)[x] for x in dict(config.__dict__) if not x.startswith('_')}
    json.dump(model_config, f)

with open(os.path.join(config.SAVE_PATH + '/config.json'), 'r') as f:
    model_config = json.load(f)

print('[INFO] Training config', json.dumps(model_config, indent=3, sort_keys=True))

 # === DATA LOADING ===
    
data_csv = pd.read_csv(config.CSV_PATH)

# train_transforms = get_augmentation_list(input_size=config.INPUT_DIM)
# train_dataset = ImageDataset(data_csv,
#                             config.IAMGES_PATH,
#                             transform=train_transforms)
train_dataset = DummyDataset(input_size=config.INPUT_DIM, num_samples=5000, channels_num=3)
train_loader = DataLoader(train_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=os.cpu_count(),
                          pin_memory=True if device == 'cuda' else False)


# ===  MODEL SETUP ===


backbone, backbone_dim = get_backbone(config.BACKBONE_NAME, config.BACKBONE_PARAMS)
pooling = get_pooling(config.POOLING_NAME, config.POOLING_PARAMS)
head = get_head(config.HEAD_NAME, config.HEAD_PARAMS)

model = Model(config.CLASS_NUM, backbone, pooling, head, embed_dim=config.EMBED_DIM, backbone_dim=backbone_dim).to(device)


# === TRAINING SETUP ===

criterion = nn.CrossEntropyLoss()
schedule = get_scheduler(config.SCHEDULER_NAME, config.BATCH_SIZE, config.SCHEDULER_PARAMS)

optimizer = optim.Adam(model.parameters(), lr=schedule(0))

if config.IS_RESUME:
    model = torch.load(config.LOAD_PATH).to(device)
    print('[INFO] Model loaded from checkpoint', json.dumps(model_config, indent=3, sort_keys=True))

if DISTRIBUTED:
    model = torch.nn.DataParallel(model)

# === START TRAINING

trainer = Trainer(criterion=criterion,
                  optimizer=optimizer,
                  device=device,
                  start_epoch=config.START_EPOCH,
                  mixed_presicion=config.MIXED_PRESICION)

print("[INFO] training the network...")
trainer.run(model, train_loader,
            epochs=config.EPOCHES_NUM,
            save_path=config.SAVE_PATH,
            schedule=schedule,
            mixed_presicion=config.MIXED_PRESICION)

