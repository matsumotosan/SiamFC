import torch
import torch.nn as nn
import torch.functional as F


class SiameseNet(nn.Module):
    def __init__(self, embedding_net) -> None:
        super().__init__()
        self.embedding_net = embedding_net
    
    def forward(self, z, x):
        return self._xcorr(self.embedding_net(z), self.embedding_net(x))
    
    def _xcorr(self, z, x):
        """Calculates cross correlation of target image and search iamge."""
        return None