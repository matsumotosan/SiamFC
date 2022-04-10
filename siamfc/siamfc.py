import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import create_labels, xcorr
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np



class SiamFCNet(pl.LightningModule):
    def __init__(self, encoder,epoch_num, batch_size, initial_lr,ultimate_lr, loss, output_scale=0.001, pretrained=False):
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
        self.cuda = torch.cuda.is_available()
        print(self.cuda)
        #self._device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.encoder = encoder
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.ultimate_lr = ultimate_lr
        self.epoch_num = epoch_num
        self.gamma = np.power(self.ultimate_lr/self.initial_lr,1/self.epoch_num)
        self.loss = loss
        self._init_weights()
        
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
        loss, center_error = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log('center_error',center_error)
        # result = pl.TrainResult(minimize=loss, on_epoch=True)
        # result.log('train_loss', loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Returns loss for validation step."""
        loss, center_error = self._shared_step(batch, batch_idx)
        self.log('train_loss',loss)
        self.log('center_error',center_error)
        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('avg_val_loss', loss)
        return {"val_loss": loss}

    def configure_optimizers(self): 
        """Returns optimizer for model."""
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.initial_lr)
        schedular = ExponentialLR(optimizer,self.gamma)
        return [optimizer],[schedular]

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
        responses_np = responses.detach().cpu().numpy()
        center_error = self.center_error(responses_np,self.total_stride)
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
        
        return loss,center_error
    
    def _xcorr(self,hz,hx):
        nz = hz.size(0)
        nx, c, h, w = hx.size()
        hx = hx.view(-1,nz*c,h,w)
        out = F.conv2d(hx,hz,groups=nz)
        out = out.view(nx,-1,out.size(-2),out.size(-1))
        return out
    
    def center_error(self,output, upscale_factor):
        """This metric measures the displacement between the estimated center of the target and the ground-truth 
        
        Args:
            output: (np.ndarray) The output of the network with dimension [Bx1xHxW]
            upscale_factor: (int) Indicates how much we must upscale the output feature map to match it to he input images
        
        Returns:
            c_error:(int) The center displacement in pixels
        """
        b = output.shape[0]
        s = output.shape[-1]
        out_flat = output.reshape(b,-1)
        max_idx = np.argmax(out_flat,axis=1)
        estim_center = np.stack([max_idx//s, max_idx%s],axis=1)
        dist = np.linalg.norm(estim_center-s//2,axis=1)
        c_error = dist.mean()
        c_error = c_error*upscale_factor
        return c_error
    
        
        