import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import create_labels, xcorr


class SiamFCNet(pl.LightningModule):
    def __init__(self, encoder, batch_size, lr, loss, output_scale=0.001, pretrained=False):
        """Fully-convolutional Siamese architecture.
         
        Calculates score map of similarity between embeddings of exemplar images (z)
        with respect to search images (x).
        
        Parameters
        ----------
        encoder : nn.Module
            Encoding network to embed exemplar and search images
        
        batch_size : int
            Batch size
        
        lr : float
            Learning rate
         
        loss : 
            Loss function
        
        output_scale : float, default=0.01
            Output scaling factor for response maps
        
        pretrained : bool, default=False
            Option to use pretrained encoder network
        """
        super().__init__()
        self.encoder = encoder
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        # self._init_weights()
        
        self.output_scale = output_scale
        #self.output_stride = self.encoder.output_stride
        self.r_pos = 16
        self.r_neg = 0
        self.total_stride = self.encoder.total_stride
        
    def forward(self, z, x):
        """Calculate response map for pairs of exemplar and search images.
        
        Parameters
        ----------
        z : array of shape (N, 3, Wz, Hz)
            Exemplar image
            
        x : array of shape (N, 3, Wx, Hx)
            Search image
            
        Returns
        -------
        response_map : array of shape (N, 1, Wr, Hr)
            Cross-correlation response map of embedded images
        """
        exemplar_embedded = self.encoder(z)
        search_embedded = self.encoder(x)
        return self._xcorr(exemplar_embedded, search_embedded) * self.output_scale    

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
        # Encode exemplar and search images
        (z, x) = batch
        hz = self.encoder(z)
        hx = self.encoder(x)
        
        # Calculate cross-correlation response map
        responses = self._xcorr(hz, hx) * self.output_scale
        
        # Generate ground truth score map
        if not (hasattr(self, 'labels') and self.labels.size() == responses.size()):
            labels = create_labels(
                responses.size(),
                self.total_stride,
                self.r_pos,
                self.r_neg
            )
            self.labels = torch.from_numpy(labels).to(self.device).float()
        
        # Calculate loss
        loss = self.loss(responses, self.labels)
        
        return loss
    
    def _xcorr(self, z, x):
        """Calculates cross-correlation between exemplar exemplar and search image embeddings.
    
        Parameters
        ----------
        z : ndarray of shape (N, C, Hz, Wz)
            Exemplar images embeddings
        
        x : ndarray of shape (N, C, Hx, Wx)
            Search images embeddings
        
        scale_factor: int, default=None
            Upsampling scaling factor (same in all spatial dimensions)
            Bertinetto et al. set to 16 during tracking (17, 17) -> (272, 272)
            Can be set to 'None' (implicitly equal to 1) during training
        
        mode : str, default='bicubic'
            Upsampling interpolation method
            Choose from {'linear', 'bilinear', 'bicubic', 'trilinear', False}.
        
        Returns
        -------
        score_map : ndarray of shape (N, 1, Hmap * scale_factor, Wmap * scale_factor)
            Score map
            
        References
        ----------
        https://pytorch.org/docs/stable/generated/torch.nn.functional.upsample.html#torch.nn.functional.upsample
        """
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out