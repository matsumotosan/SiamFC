import torch
import torch.nn as nn
import torch.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.init_weights()
        
    def init_weights(self):
        pass
    
    def forward(self, x):
        z = self.model(x)
        return z


class AlexNet(nn.Module):
    pass


class ContrastiveRandomWalkNet(nn.Module):
    pass


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.weight)

def load_pretrained():
    pass