from glob import glob
import xdrlib
import torch
from torch import nn
import torch.nn.functional as F
from arc_triplet_loss import *

    
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Model(nn.Module):
    def __init__(self, backbone, pooling, head, embed_dim=512, backbone_dim=2048, arc_criterion=nn.CrossEntropyLoss()):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pool = pooling
        self.head = head
        self.arc_criterion = arc_criterion
        
        self.arc_fc = nn.Linear(backbone_dim, embed_dim)
        # nn.init.normal_(self.arc_fc.weight, std=0.001)
        # nn.init.constant_(self.arc_fc.bias, 0)
        
        self.local_conv = nn.Conv2d(backbone_dim, embed_dim, 1)
        self.local_bn = nn.BatchNorm2d(embed_dim)
        self.local_bn.bias.requires_grad_(False)  # no shift
        self.bottleneck_g = nn.BatchNorm1d(backbone_dim)
        self.bottleneck_g.bias.requires_grad_(False)  # no shift
        

    
        
    def forward(self, x, targets = None):
        x = self.backbone(x)
        # x = torch.squeeze(torch.squeeze(self.pool(x), -1), -1)

            
        # Global feats
        
        # global_feat = F.avg_pool2d(x, x.size()[2:])
        global_feat = torch.squeeze(torch.squeeze(self.pool(x), -1), -1)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        # local feat
        local_feat = torch.mean(x, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)
        
        
        arc_feats = F.relu(self.arc_fc(global_feat))
        arc_feats = l2_norm(arc_feats)
        
        if targets is not None:
            logits = self.head(arc_feats, targets)
            return global_feat, local_feat, logits
        
        return global_feat, local_feat, arc_feats
    
    def get_loss(self, global_feat, local_feat, logits, targets):
        triplet_loss = global_loss(TripletLoss(margin=0.3), global_feat, targets)[0] + \
                      local_loss(TripletLoss(margin=0.3), local_feat, targets)[0]
                      
        arc_loss = self.arc_criterion(logits, targets)
        
        return triplet_loss + arc_loss

if __name__ == '__main__':
    from zoo import get_backbone,\
                        get_pooling, \
                        get_head

    backbone, backbone_dim = get_backbone('effnetv1_b2')
    pool = get_pooling('gem')
    head = get_head('adacos', head_params={'feat_dim':512, 'num_classes':15567})
    
    model = Model(backbone, pool, head, embed_dim=512, backbone_dim=backbone_dim)
    x = torch.randn([4, 3, 260, 260])
    targets = torch.tensor([0, 0, 2, 3])
    gf, lf, logits = model(x, targets)    
    loss = model.get_loss(gf, lf, logits, targets)
    print(loss)
    
    