from torch import nn
from torchvision.models import efficientnet_b2, \
                                efficientnet_b3, \
                                efficientnet_b4, \
                                efficientnet_b5, \
                                efficientnet_b6, \
                                efficientnet_b7, \
                                efficientnet_v2_s, \
                                efficientnet_v2_m, \
                                efficientnet_v2_l

from torchvision.models import resnext50_32x4d, resnext101_32x8d
from .backbones import *
from .head import *
from .pool import *
from .schedulers import *

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
        
    elif backbone_name == 'effnetv2_s':
        backbone = nn.Sequenital(*(list(efficientnet_v2_s(**backbone_params).children())[:-2]))
        backbone_dim = 1280
        
    elif backbone_name == 'effnetv2_m':
        backbone = nn.Sequential(*(list(efficientnet_v2_m(**backbone_params).children())[:-2]))
        backbone_dim = 1280
        
    elif backbone_name == 'effnetv2_l':
        backbone = nn.Sequential(*(list(efficientnet_v2_l(**backbone_params).children())[:-2]))
        backbone_dim = 1280
        
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
    
    
def get_scheduler(scheduler_name='ramp', scheduler_params={}):
    
    if scheduler_name == 'ramp':
        schduler = LrRampScheduler(**scheduler_params)
    elif scheduler_name == 'decay':
        schduler = LrDecayScheduler(**scheduler_params)
    
    else:
        assert False, f'Error, unknown pooling name {scheduler_name}'
            
    return schduler


