from utils.trainer import Trainer
from configs.fast_train import config
import torch
from torch import nn, optim
import pandas as pd
import os
import json
from utils import  set_seed, Augmenter
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Convert
from ffcv.loader import Loader, OrderOption
from modules.factory import get_backbone,\
                        get_pooling, \
                        get_head, \
                        get_scheduler

from modules import Model
from modules.arc_triplet_loss import ArcTripletLoss
from torch.nn.parallel import DistributedDataParallel

set_seed(config.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    NUM_GPU = torch.cuda.device_count()
    print(f'[INFO] number of GPUs found: {NUM_GPU}')
    if NUM_GPU > 1:
        DISTRIBUTED = True
        config.BATCH_SIZE = config.BATCH_SIZE * NUM_GPU
        rank = NUM_GPU - 1
        this_device = f'cuda:{rank}'
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
    

pipelines = {
    'image': [SimpleRGBImageDecoder(), ToTensor(), Convert(torch.float32), ToDevice(torch.device(this_device)), ToTorchImage(), Augmenter()],
    'label': [IntDecoder(), ToDevice(torch.device(this_device))]
}

train_loader = Loader(config.DATA_PATH, batch_size=config.BACKBONE_PARAMS, num_workers=os.cpu_count()-1,
                order=OrderOption.SEQUENTIAL, pipelines=pipelines, os_cache=True, distributed=True)


# ===  MODEL SETUP ===


backbone, backbone_dim = get_backbone(config.BACKBONE_NAME, config.BACKBONE_PARAMS)
pooling = get_pooling(config.POOLING_NAME, config.POOLING_PARAMS).to(device)
head = get_head(config.HEAD_NAME, config.HEAD_PARAMS).to(device)

model = Model(backbone, pooling, head, embed_dim=config.EMBED_DIM, backbone_dim=backbone_dim).to(device)


# === TRAINING SETUP ===

criterion = ArcTripletLoss(margin=config.TRIPLET_MARGIN)
schedule = get_scheduler(config.SCHEDULER_NAME, config.BATCH_SIZE, config.SCHEDULER_PARAMS)

optimizer = optim.AdamW(model.parameters(), lr=schedule(0), weight_decay=1e-5)

if config.IS_RESUME:
    model = torch.load(config.LOAD_PATH).to(device)
    print('[INFO] Model loaded from checkpoint', json.dumps(model_config, indent=3, sort_keys=True))

if DISTRIBUTED:
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model).to(rank)


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

