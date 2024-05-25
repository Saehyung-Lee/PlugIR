import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from torch import nn

def Contrastive(img_features, txt_features, temp, device):
    ###============== Image-text Contrastive Learning ===================###
    xent = nn.CrossEntropyLoss()
    targets = torch.arange(img_features.size(0), device=device)

    sim_i2t = img_features @ txt_features.t() / temp
    sim_t2i = txt_features @ img_features.t() / temp 
    
    loss_i2t = xent(sim_i2t, targets)
    loss_t2i = xent(sim_t2i, targets)

    return (loss_i2t+loss_t2i)/2

class RecallatK(torch.nn.Module):
    def __init__(self):
        super(RecallatK, self).__init__()
        self.temp = 0.01
        self.k_vals=[1,2,4,8,16]

    def _get_offdiag(self, x):
        return x[~torch.eye(x.size(0), dtype=torch.bool)].view(x.size(0), x.size(1)-1)

    def forward(self, img_features, txt_features):
        sims = txt_features @ img_features.t()
        ranks = 1 + torch.sigmoid(self._get_offdiag(sims - sims.diagonal().unsqueeze(1))  / self.temp).sum(1)
        loss = 0.
        for k in self.k_vals:
            Rk = torch.sigmoid(k-ranks).mean()
            Lk = 1 - Rk
            loss = loss + Lk / len(self.k_vals)
        return loss
