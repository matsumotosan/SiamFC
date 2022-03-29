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
    """AlexNet architecture as specified by Bertinetto et al. (2016).
    
    References
    ----------
    Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, and Philip H. S. Torr. 
    Fully-Convolutional Siamese Networks for Object Tracking. 6 2016.
    """
    def __init__(self) -> None:
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 96, 11, 2, groups=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            # Layer 2
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            # Layer 3
            nn.Conv2d(256, 384, 3, 1, groups=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Layer 5
            #nn.Conv2d(192, 256, 3, 1, groups=2)
            nn.Conv2d(384, 256, 3, 1, groups=2)
            )
    
    def forward(self, x):
        return self.model(x)
    
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_pretrained():
    pass