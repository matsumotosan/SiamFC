import torch
import torch.nn as nn
from collections import OrderedDict


class AlexNet(nn.Module):
    """AlexNet encoder architecture as specified by Bertinetto et al. (2016).
    
    References
    ----------
    Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, and Philip H. S. Torr. 
    Fully-Convolutional Siamese Networks for Object Tracking. 6 2016.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2, groups=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, groups=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )
        self.total_stride = 8
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def load_pretrained(self, file) -> None:
        """Load pretrained network for encoder
        
        Weights are from https://github.com/huanglianghua/siamfc-pytorch.
        
        Parameters
        ----------
        file : str
            File containing pretrained network parameters
        """
        # Load state_dict
        state_dict = torch.load(file, map_location=torch.device('cpu'))
       
        # Rename state_dict keys to match our model
        new_state_dict = OrderedDict() 
        for (k, v) in state_dict.items():
            new_k = k[9:]
            new_state_dict[new_k] = v
        
        # Load weights using new state_dict
        self.load_state_dict(new_state_dict)


class AlexNet_torch(nn.Module):
    def __init__(self,pretrained=False):
        super(AlexNet_torch,self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pretrained)
        self.total_stride = 8
            
    def modify(self,padding=''):
        #remove the average pooling layer and the classifier
        remove_layers = ['avgpool','classifier']
        for layer in remove_layers:
            setattr(self.model,layer,None)
        #remove the last relu activation and pooling layer 
        self.model.features[12] = None
        self.model.features[11] = None
        #set the stride of the first conv layer to be 2
        self.model.features[0].stride = tuple(2 for _ in self.model.features[0].stride)
        #modify the padding if specified
        if padding != '' and padding != 'no':
            for m in self.model.modules():
                if isinstance(m, torch.nn.Conv2d) and sum(m.padding) > 0:
                    m.padding_mode = padding
        elif padding == 'no':
            for m in self.model.modules():
                if isinstance(m, torch.nn.Conv2d) and sum(m.padding) > 0:
                    m.padding = (0,0)
        
    def forward(self, x):
        for i in range(11):
            x = self.model.features[i](x)
        return x