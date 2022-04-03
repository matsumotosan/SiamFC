import torch
import torch.nn as nn
import torch.functional as F
from pytorch_lightning import LightningModule
from .losses import bce_loss


class SiamFCNet(LightningModule):
    def __init__(self, embedding_net, output_scale=0.001) -> None:
        super().__init__()
        self.embedding_net = embedding_net
        self.output_scale = output_scale
    
    def init_weights(self) -> None:
        """Initialize weights of embedding network"""
        self.embedding_net.init_weights()
    
    def forward(self, z, x):
        """Forward pass to calculate score map for search and target images."""
        target_embedded = self.embedding_net(z)
        search_embedded = self.embedding_net(x)
        return self._xcorr(target_embedded, search_embedded)    

    def training_step(self, batch, batch_nb):
        """Calculates loss for one step in training."""
        x, y = batch
        loss = bce_loss(self(x), y)
        return loss

    def configure_optimizer(self):
        """Returns optimizer for model."""
        lr = self.hparams.lr
        opt = torch.optim.Adam(self.embedding_net.parameters(), lr=lr)
        return opt

    def _xcorr(z, x):
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