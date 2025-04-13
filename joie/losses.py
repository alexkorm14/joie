# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torchkge.utils import MarginLoss


class MarginLoss_CVG(nn.Module):
    """Description.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, margin):
        super().__init__()
        self.loss = nn.MarginRankingLoss(margin=-margin, reduction='sum')
        
    def forward(self, positive_triplets, negative_triplets):
        
        return self.loss(torch.zeros_like(negative_triplets),positive_triplets,target=torch.ones_like(positive_triplets))


class MarginLoss_CVT(nn.Module):
    """Description.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, margin):
        super().__init__()
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='sum')

    def forward(self, positive_triplets, negative_triplets):
       return self.loss(negative_triplets,positive_triplets,target=torch.ones_like(positive_triplets))
