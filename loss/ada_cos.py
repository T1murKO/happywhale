import torch
from torch import nn
import math
import torch.nn.functional as F


class AdaCos(nn.Module):
    def __init__(self, feat_dim, num_classes, fixed_scale=False):
        super(AdaCos, self).__init__()
        self.fixed_scale = fixed_scale
        self.scale = math.sqrt(2) * math.log(num_classes - 1)
        self.W = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, feats, labels):
        W = F.normalize(self.W)

        logits = F.linear(feats, W)

        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        if self.fixed_scale:
            with torch.no_grad():
                B_avg = torch.where(one_hot < 1, torch.exp(self.scale * logits), torch.zeros_like(logits))
                B_avg = torch.sum(B_avg) / feats.size(0)
                
                theta_med = torch.median(theta[one_hot == 1])
                self.scale = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
            
        output = self.scale * logits
        return output
    
    def get_logits(self, feats):
        W = F.normalize(self.W)

        logits = F.linear(feats, W)
        return logits
