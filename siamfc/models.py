import torch
import torch.nn as nn
import torch.functional as F


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
            nn.Conv2d(384, 256, 3, 1, groups=2)
            )
    
    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, file, freeze=False) -> None:
        """Load pretrained network for encoder
        
        Parameters
        ----------
        file : str
            File containing pretrained network parameters
        """
        self.model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
        if freeze:
            self.model.freeze()

class ContrastiveRandomWalkNet(nn.Module):
    pass