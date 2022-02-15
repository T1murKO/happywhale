from torchvision.models import efficientnet_b5
from torch import nn, optim
from loss import ArcFace
from utils import ramp_scheduler
import torch
from models import Model
from utils import GeM
from utils import get_augmentation_list

# EffNetB0	224
# EffNetB1	240
# EffNetB2	260
# EffNetB3	300
# EffNetB4	380
# EffNetB5	456
# EffNetB6	528
# EffNetB7	600

class Config:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    csv_path = ''
    images_path = ''
    # Model settings
    
    embed_dim = 512
    input_dim = (456, 456)
    class_num = 15000
    
    num_epoch = 20
    start_epoch = 0
    batch_size = 32
    
    
    backbone_ = efficientnet_b5
    # Remove last 2 layers of pooling and dense
    backbone = nn.Sequential(*(list(backbone_(pretrained=True).children())[:-2]))
    
    # Type of pooling
    pooling = GeM()
    
    # Classification head
    head = ArcFace(embed_dim, class_num, s=30., m=0.5)
    
    # Training loss
    loss = nn.CrossEntropyLoss()
    
    
    model = Model(class_num, backbone, pooling, head, embed_dim=embed_dim).to(device)
    
    # Training scheduler. List of learning rates
    schedule = ramp_scheduler(num_epoch, batch_size, start_epoch=start_epoch)
    
    optimizer = optim.Adam(model.parameters(), lr=schedule[0])
    
    train_transforms = get_augmentation_list(input_dim)
    
    
    # Save path
    
    save_path = '/'
    
    
    
    
    
    
    