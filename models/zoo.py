# from torchvision.models import *
# from torch import nn
# from .utils import *

# def get_backbone(backbone_name='resnext50'):
    
#         if backbone_name == 'resnext50':
#             backbone = nn.Sequential(*(list(resnext50_32x4d(pretrained=True).children())[:-2]))
#             backbone_dim = 2048
            
#         elif backbone_name == 'resnext101':
#             backbone = nn.Sequential(*(list(resnext101_32x8d(pretrained=True).children())[:-2]))
#             backbone_dim = 2048
        
#         elif backbone_name == 'effnetv1_b2':
#             backbone = nn.Sequential(*(list(efficientnet_b2(pretrained=True).children())[:-2]))
#             backbone_dim = 1408
        
#         elif backbone_name == 'effnetv1_b3':
#             backbone = nn.Sequential(*(list(efficientnet_b3(pretrained=True).children())[:-2]))
#             backbone_dim = 1536
        
#         elif backbone_name == 'effnetv1_b4':
#             backbone = nn.Sequential(*(list(efficientnet_b4(pretrained=True).children())[:-2]))
#             backbone_dim = 1792
        
#         elif backbone_name == 'effnetv1_b5':
#             backbone = nn.Sequential(*(list(efficientnet_b5(pretrained=True).children())[:-2]))
#             backbone_dim = 2048
        
#         elif backbone_name == 'effnetv1_b6':
#             backbone = nn.Sequential(*(list(efficientnet_b6(pretrained=True).children())[:-2]))
#             backbone_dim = 2304
        
#         elif backbone_name == 'effnetv1_b7':
#             backbone = nn.Sequential(*(list(efficientnet_b7(pretrained=True).children())[:-2]))
#             backbone_dim = 2560
        
#         else:
#             assert False, f'Error, unknown backbone name {backbone_name}'
            
#         return backbone, backbone_dim


# def get_pooling(pooling_name='GeM'):
#     if pooling_name == 'GeM':
#         pool = GeM()
    
#     else:
#         assert False, f'Error, unknown backbone name {pooling_name}'
        
        
    
#     return pool