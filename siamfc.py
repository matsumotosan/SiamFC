import torch
import torch.nn as nn
import torch.functional as F
from pytorch_lightning import LightningModule
from losses import *
import numpy as np

class SiamFCNet(LightningModule):
    def __init__(self, embedding_net, output_scale=0.001) -> None:
        super().__init__()
        self.embedding_net = embedding_net
        self.output_scale = output_scale
        self.init_weights()
        self.total_stride = 8
        self.r_pos = 16
        self.r_neg = 0
        self.lr = 1e-2
    
    def init_weights(self) -> None:
        """Initialize weights of embedding network"""
        #self.embedding_net.init_weights()
        for m in self.embedding_net.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight,1)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight,1)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            
    
    def forward(self, z, x):
        """Forward pass to calculate score map for search and target images."""
        target_embedded = self.embedding_net(z)
        search_embedded = self.embedding_net(x)
        return self._xcorr(target_embedded, search_embedded)*self.output_scale    

    def training_step(self, batch, batch_nb): #What does batch_nb mean?
        """Calculates loss for one step in training."""
        z = batch[0]
        x = batch[1]
        #responses = self(z,x)
        target_embedded = self.embedding_net(z)
        search_embedded = self.embedding_net(x)
        #print(target_embedded.shape,search_embedded.shape)
        responses = self._xcorr(target_embedded,search_embedded)
        labels = self._create_labels(responses.size())
        loss = bce_loss_balanced(self(z,x), labels)
        return loss

    def configure_optimizers(self): 
        """Returns optimizer for model."""
        #lr = self.hparams.lr
        #lr = 1e-2
        opt = torch.optim.Adam(self.embedding_net.parameters(), lr=self.lr)
        return opt

    def _xcorr(self,z, x):
        """Calculates cross-correlation between target and search embeddings.
        
        Parameters
        ----------
        z : ndarray of shape ()
            Target embedding
        
        x : ndarray of shape ()
            Search embedding
        
        Returns
        -------
        score_map : ndarray of shape ()
            Score map
        """
        score_map = F.conv2d(x, z)
        return score_map
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.r_pos / self.total_stride
        r_neg = self.r_neg / self.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels