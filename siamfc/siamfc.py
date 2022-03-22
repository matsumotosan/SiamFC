import torch
import torch.nn as nn
import torch.functional as F
from pytorch_lightning import LightningModule
from losses import loss

class SiameseNet(LightningModule):
    def __init__(self, embedding_net) -> None:
        super().__init__()
        self.embedding_net = embedding_net
    
    def init_weights(self) -> None:
        self.embedding_net.init_weights()
    
    def forward(self, z, x):
        target_embedded = self.embedding_net(z)
        search_embedded = self.embedding_net(x)
        return self._xcorr(target_embedded, search_embedded)    

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = loss(self(x), y)
        return loss

    def configure_optimizer(self):
        lr = self.hparams.lr
        opt = torch.optim.Adam(self.embedding_net.parameters(), lr=lr)
        return opt

    def _xcorr(z, x):
        return None