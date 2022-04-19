import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl
import numpy as np
from torchvision import transforms
from .utils import create_labels
from .metrics import calc_center_error


class SiamFCNet(pl.LightningModule):
    def __init__(
        self, 
        encoder,
        loss,
        epoch_num=50,
        batch_size=8,
        initial_lr=1e-2,
        ultimate_lr=1e-5,
        weight_decay=5e-4, 
        output_scale=0.001,
        preprocess=False,
        init_weights=False
        ):
        """Fully-convolutional Siamese architecture (SiamFC).

        Calculates score map of similarity between embeddings of exemplar images (z)
        with respect to search images (x).

        Parameters
        ----------
        encoder : nn.Module
            Encoding network to embed exemplar and search images

        loss :
            Loss function

        batch_size : int
            Batch size

        lr : float
            Learning rate

        output_scale : float, default=0.01
            Output scaling factor for response maps

        pretrained : bool, default=False
            Option to use pretrained encoder network
        """
        super().__init__()
        self.preprocess = preprocess
        self.normalize = torch.nn.Sequential(
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            )
        self.encoder = encoder
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.ultimate_lr = ultimate_lr
        self.weight_decay = weight_decay
        self.epoch_num = epoch_num
        self.loss = loss
        if init_weights == True:
            self._init_weights()

        self.output_scale = output_scale
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

    def training_step(self, batch, batch_idx):
        """Returns and logs loss for training step."""
        loss, center_error = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("center_error", center_error)
        # result = pl.TrainResult(minimize=loss, on_epoch=True)
        # result.log('train_loss', loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Returns loss for validation step."""
        loss, center_error = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("center_error", center_error)
        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('avg_val_loss', loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """Returns loss for test step."""
        loss, center_error = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("center_error", center_error)
        return {"test_loss", loss}

    def configure_optimizers(self, optimizer='sgd'): 
        """Returns optimizer for model."""
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.encoder.parameters(), 
                lr=self.initial_lr, 
                weight_decay=self.weight_decay)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.encoder.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay)

        gamma = np.power(self.ultimate_lr / self.initial_lr, 1 / self.epoch_num)
        scheduler = ExponentialLR(optimizer, gamma)
        return [optimizer], [scheduler]

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
        if self.preprocess == True:
            z /= 255
            x /= 255
            z = self.normalize(z)
            x = self.normalize(x)

        # Calculate response map
        responses = self.forward(z, x)
        responses_np = responses.detach().cpu().numpy()

        # Calculate center error
        center_error = calc_center_error(responses_np, self.total_stride)

        # Generate ground truth score map
        if not (hasattr(self, 'labels') and self.labels.size() == responses.size()):
            labels = create_labels(
                responses.size(),
                self.total_stride,
                self.r_pos,
                self.r_neg
            )
            self.labels = torch.from_numpy(labels).to(self.device).float()

        # Calculate loss (BCE or triplet)
        loss = self.loss(responses, self.labels)
        return loss, center_error

    def _xcorr(self, hz, hx):
        nz = hz.size(0)
        nx, c, h, w = hx.size()
        hx = hx.view(-1, nz * c, h, w)
        out = F.conv2d(hx, hz, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out