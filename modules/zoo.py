from torch import nn
from torchvision.models import efficientnet_b2, \
                                efficientnet_b3, \
                                efficientnet_b4, \
                                efficientnet_b5, \
                                efficientnet_b6, \
                                efficientnet_b7

from torchvision.models import resnext50_32x4d, resnext101_32x8d
from modules.senet import *
from modules.head import *
from modules.pool import *
from utils import ramp_scheduler

def get_backbone(backbone_name='resnext50', backbone_params={}):
    
    if backbone_name == 'resnext50':
        backbone = nn.Sequential(*(list(resnext50_32x4d(**backbone_params).children())[:-2]))
        backbone_dim = 2048
        
    elif backbone_name == 'resnext101':
        backbone = nn.Sequential(*(list(resnext101_32x8d(**backbone_params).children())[:-2]))
        backbone_dim = 2048
        
    elif backbone_name == 'se_resnext50':
        backbone = se_resnext50_32x4d(**backbone_params)
        backbone_dim = 2048
    
    elif backbone_name == 'senet154':
        backbone = senet154(**backbone_params)
        backbone_dim = 2048
    
    elif backbone_name == 'effnetv1_b2':
        backbone = nn.Sequential(*(list(efficientnet_b2(**backbone_params).children())[:-2]))
        backbone_dim = 1408
    
    elif backbone_name == 'effnetv1_b3':
        backbone = nn.Sequential(*(list(efficientnet_b3(**backbone_params).children())[:-2]))
        backbone_dim = 1536
    
    elif backbone_name == 'effnetv1_b4':
        backbone = nn.Sequential(*(list(efficientnet_b4(**backbone_params).children())[:-2]))
        backbone_dim = 1792
    
    elif backbone_name == 'effnetv1_b5':
        backbone = nn.Sequential(*(list(efficientnet_b5(**backbone_params).children())[:-2]))
        backbone_dim = 2048
    
    elif backbone_name == 'effnetv1_b6':
        backbone = nn.Sequential(*(list(efficientnet_b6(**backbone_params).children())[:-2]))
        backbone_dim = 2304
    
    elif backbone_name == 'effnetv1_b7':
        backbone = nn.Sequential(*(list(efficientnet_b7(**backbone_params).children())[:-2]))
        backbone_dim = 2560
    
    elif backbone_name == 'effnetv1_b7':
        backbone = nn.Sequential(*(list(efficientnet_b7(**backbone_params).children())[:-2]))
        backbone_dim = 2560
    else:
        assert False, f'Error, unknown backbone name {backbone_name}'
        
    return backbone, backbone_dim
    
    

def get_head(head_name='adacos', head_params={}):
    
    if head_name == 'adacos':
        head = AdaCos(**head_params)

    elif head_name == 'arcface':
        head = ArcFace(**head_params)
        
    else:
        assert False, f'Error, unknown head name {head_name}'
            
    return head


def get_pooling(pool_name='gem', pool_params={}):
    
    if pool_name == 'gem':
        pool = GeM(**pool_params)

    else:
        assert False, f'Error, unknown pooling name {pool_name}'
            
    return pool
    
    
def get_scheduler(scheduler_name='ramp', batch_size=32, scheduler_params={}):
    
    if scheduler_name == 'ramp':
        schduler = ramp_scheduler(batch_size, **scheduler_params)

    else:
        assert False, f'Error, unknown pooling name {pool_name}'
            
    return schduler