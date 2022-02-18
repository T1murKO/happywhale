import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, backbone, pooling, head, embed_dim=512, backbone_dim=2048):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pool = pooling
        self.head = head
        
        self.bn1 = nn.BatchNorm1d(backbone_dim)
        self.fc1 = nn.Linear(backbone_dim, embed_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, targets = None):
        x = self.backbone(x)
        x = torch.squeeze(torch.squeeze(self.pool(x), -1), -1)

        x = F.relu(self.fc1(self.dropout(self.bn1(x))))
        x = F.normalize(x)
        
        if targets is not None:
            logits = self.head(x, targets)
            return logits

        return x
    
    # def get_logits(self, x):
    #     x = self.gem_pool(self.backbone(x))
    #     x = torch.unsqueeze(torch.squeeze(x), 0)
    #     x = F.relu(self.fc1(self.dropout(self.bn1(x))))
    #     x = F.normalize(x)

    #     logits = self..get_logits(x)
    #     return logits
    