import torch
import torch.nn as nn


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
            

# class _BatchNorm2d(nn.BatchNorm2d):

#     def __init__(self, num_features, *args, **kwargs):
#         super(_BatchNorm2d, self).__init__(
#             num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


# class _AlexNet(nn.Module):
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         return x


# class AlexNetV1(_AlexNet):
#     output_stride = 8

#     def __init__(self):
#         super(AlexNetV1, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 96, 11, 2),
#             _BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2))
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(96, 256, 5, 1, groups=2),
#             _BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2))
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(256, 384, 3, 1),
#             _BatchNorm2d(384),
#             nn.ReLU(inplace=True))
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(384, 384, 3, 1, groups=2),
#             _BatchNorm2d(384),
#             nn.ReLU(inplace=True))
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(384, 256, 3, 1, groups=2))
        
#         self.total_stride = 8