from utils.trainer import Trainer
from configs.progressive_train_1 import config
import torch
from torch import nn, optim
import os
import json
from utils import set_seed
from utils.transforms import ProgressiveAugmenter
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Convert
from ffcv.loader import Loader, OrderOption
from modules.factory import get_backbone,\
                        get_pooling, \
                        get_head, \
                        get_scheduler
                        
from modules.schedulers import WeightDecayScheduler

from modules import Model
from modules.arc_triplet_loss import ArcTripletLoss
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

set_seed(config.SEED)


def setup_distributed_device(gpu_rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=gpu_rank, world_size=world_size)

    torch.cuda.set_device(gpu_rank)


def train(gpu_rank, world_size):
    setup_distributed_device(gpu_rank, world_size)
    
    this_device = f'cuda:{gpu_rank}'
    
    augmenter = ProgressiveAugmenter(config.AUG_SETTINGS)
    
    pipelines = {
    'image': [SimpleRGBImageDecoder(), ToTensor(), Convert(torch.float32), ToDevice(torch.device(this_device)), ToTorchImage(), augmenter],
    'label': [IntDecoder(), ToDevice(torch.device(this_device))]
    }

    train_loader = Loader(config.DATA_PATH, batch_size=config.BACKBONE_PARAMS, num_workers=os.cpu_count()-1,
                order=OrderOption.SEQUENTIAL, pipelines=pipelines, os_cache=True, distributed=True)
    
    
    backbone, backbone_dim = get_backbone(config.BACKBONE_NAME, config.BACKBONE_PARAMS)
    pooling = get_pooling(config.POOLING_NAME, config.POOLING_PARAMS).to(device)
    head = get_head(config.HEAD_NAME, config.HEAD_PARAMS).to(device)

    model = Model(backbone, pooling, head, embed_dim=config.EMBED_DIM, backbone_dim=backbone_dim)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[gpu_rank])
    
    criterion = ArcTripletLoss(margin=config.TRIPLET_MARGIN)
    lr_scheduler = get_scheduler(config.SCHEDULER_NAME, config.BATCH_SIZE, config.SCHEDULER_PARAMS)
    wd_scheduler = WeightDecayScheduler()
    
    
    optimizer = optim.AdamW(model.parameters(), lr=lr_scheduler(config.START_EPOCH), weight_decay=wd_scheduler(config.START_EPOCH))

    trainer = Trainer(criterion=criterion,
                    optimizer=optimizer,
                    device=gpu_rank,
                    start_epoch=config.START_EPOCH,
                    lr_scheduler=lr_scheduler,
                    wd_scheduler=wd_scheduler)

    
    trainer.run(model, train_loader,
                epochs=config.EPOCHES_NUM,
                save_path=config.SAVE_PATH,
                lr_scheduler=lr_scheduler)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    assert device.type != 'cuda', 'No GPU Asselerator detected'
    NUM_GPU = torch.cuda.device_count()
    # assert NUM_GPU == 1, '1 GPU detected, can\'t start distributed training'
    
    print(f'[INFO] number of GPUs found: {NUM_GPU}')
    
    if not os.path.exists(config.SAVE_PATH):
        os.mkdir(config.SAVE_PATH)
        
    with open(os.path.join(config.SAVE_PATH + '/config.json'), 'w') as f:
        model_config = {x:dict(config.__dict__)[x] for x in dict(config.__dict__) if not x.startswith('_')}
        json.dump(model_config, f)
        
    with open(os.path.join(config.SAVE_PATH + '/config.json'), 'r') as f:
        model_config = json.load(f)
    
    config.BATCH_SIZE = config.BATCH_SIZE * NUM_GPU
    
    print('[INFO] Training config', json.dumps(model_config, indent=3, sort_keys=True))
    
    
    mp.spawn(
        train,
        args=(NUM_GPU),
        nprocs=NUM_GPU
    )
