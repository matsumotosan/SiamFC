import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class SiamFCNet(pl.LightningModule):
    def __init__(self, encoder, batch_size, lr, loss, output_scale=0.001, pretrained=False):
        """Build SiamFC network
        
        Parameters
        ----------
        encoder : nn.Module
            Encoding network to embed target and search images
        
        batch_size : int
            Batch size
        
        lr : float
            Learning rate
            
        loss : 
            Loss function
            
        output_scale : float, default=0.01
            Output scaling factor for response maps
        """
        super().__init__()
        self.encoder = encoder
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self._init_weights()
        
        self.output_scale = output_scale
        self.total_stride = 8
        self.r_pos = 16
        self.r_neg = 0

    def forward(self, z, x):
        """Calculate response map for pair of exmplar and search images.
        
        Parameters
        ----------
        z : array of shape ()
            Exemplar image
            
        x : array of shape ()
            Search image
            
        Returns
        -------
        response_map : array of shape ()
            Cross-correlation response map of embedded images
        """
        target_embedded = self.encoder(z)
        search_embedded = self.encoder(x)
        return self._xcorr(target_embedded, search_embedded) * self.output_scale    

    def training_step(self, batch, batch_idx): #What does batch_nb mean?
        """Returns and logs loss for training step."""
        loss = self._shared_step(batch, batch_idx)
        # result = pl.TrainResult(minimize=loss, on_epoch=True)
        # result.log('train_loss', loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Returns loss for validation step."""
        loss = self._shared_step(batch, batch_idx)
        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('avg_val_loss', loss)
        return {"val_loss": loss}

    def configure_optimizers(self): 
        """Returns optimizer for model."""
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        return optimizer

    def _init_weights(self) -> None:
        """Initialize weights of encoder network."""
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _shared_step(self, batch, batch_idx):
        """Returns loss for pass through model with provided batch."""
        (z, x) = batch
        
        # Encode target and search images
        hz = self.encoder(z)
        hx = self.encoder(x)
        
        #print(hz.shape, hx.shape)
        
        # Calculate response map and loss
        responses = self._xcorr(hz, hx)
        labels = self._create_labels(responses.size())
        loss = self.loss(responses, labels)
        return loss

    def _xcorr(self, z, x):
        """Calculates cross-correlation between target and search image embeddings.
        
        Parameters
        ----------
        z : ndarray of shape (B, C, H, W)
            Target embedding
        
        x : ndarray of shape (B, C, Hx, Wx)
            Search embedding
        
        Returns
        -------
        score_map : ndarray of shape ()
            Score map
        """
        nz = z.shape[0]
        nx, cx, hx, wx, = x.shape
        x = x.view(-1, nz * cx, hx, wx)
        score_map = F.conv2d(x, z, groups=nz)
        score_map = score_map.view(nx, -1, score_map.shape[-2], score_map.shape[-1])
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
