import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Optional


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, transform, batch_size: int = 32) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def prepare_data(self, stage: Optional[str] = None) -> None:
        """Download and preprocess data."""
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup dataloader and transforms."""
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
    
    def train_dataloader(self) -> DataLoader:
        """Return dataloader for training."""
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Return dataloader for validation."""
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Return dataloader for testing."""
        return DataLoader(self.test_ds, batch_size=self.batch_size)


class Pair(Dataset):
    def __init__(self, seqs, transforms=None, pairs_per_seq=1):
        """Data class for generating target and exemplar images from sequence of video frames.
        
        Parameters
        ----------
        seqs : object
            Object containing list of filenames for raw images and annotations
            
        transforms : torch.Transform
            PyTorch transforms to be used for target and exemplar images
            
        pairs_per_seq : int, default=1
            Number of target/exemplar pairs to be generated from each sequence of video frames
        """
        super().__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        indices = np.random.permutation(len(seqs))
        self.indices = indices[indices != 331] # We need to avoid the 332th video sequence because it's corrupted
        self.return_meta = getattr(seqs, 'return_meta', False)
    
    def  __getitem__(self, index):
        index = self.indices[index % len(self.indices)]

        # Get image filenames and annotations for video
        img_files, anno  = self.seqs[index]
        # img_files, anno, _meta  = self.seqs[index]
        frame_indices = list(range(len(img_files)))
        
        # Select frame indices (ensure within maximum separation of number of frames)
        rand_z, rand_x = np.sort(np.random.choice(frame_indices,2,replace=False))
        while rand_x - rand_z > 100: 
            rand_z, rand_x = np.sort(np.random.choice(frame_indices,2,replace=False))# The two chosen frames should be at most T frames apart 
        
        # Read exemplar and target images
        z = cv2.imread(img_files[rand_z],cv2.IMREAD_COLOR) #May need to be converted to RGB color space
        x = cv2.imread(img_files[rand_x],cv2.IMREAD_COLOR) #May need to be converted to RGB color space
        
        # Get annotations for exemplar and target images
        box_z = anno[rand_z]
        box_x = anno[rand_x]
        
        # Perform image transforms on exemplar and target images
        item = (z, x, box_z, box_x)
        if self.transforms:
            z, x = self.transforms(*item)
            # item = self.transforms(*item)

        return z, x
    
    def __len__(self):
        return len(self.indices) *  self.pairs_per_seq