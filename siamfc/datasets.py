"""DataModule class for GOT-10k dataset."""
import cv2 as cv
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from got10k.datasets import GOT10k
from .transforms import SiamFCTransforms


class GOT10kDataModule(pl.LightningDataModule):
    """PyTorch LightningDataModule class for GOT-10k dataset."""
    def __init__(self, data_dir='./data/GOT-10k', batch_size=8, transform=SiamFCTransforms) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
 
    def prepare_data(self) -> None:
        """Download and prepare data."""
        pass
    
    def setup(self,  stage: Optional[str] = None) -> None:
        """Define transforms and data splits."""
        if stage == "fit" or stage is None:
            train_seqs = GOT10k(root_dir=self.data_dir, subset='train')
            val_seqs = GOT10k(root_dir=self.data_dir, subset='val')
            self.got10k_train = Pair(seqs=train_seqs, transforms=self.transform)
            self.got10k_val = Pair(seqs=val_seqs, transforms=self.transform)

        if stage == "test" or stage is None:
            test_seqs = GOT10k(root_dir=self.data_dir, subset='test')
            self.got10k_test = Pair(seqs=test_seqs, transforms=self.transform)

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        got10k_train = DataLoader(
            self.got10k_train, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=12
        )
        return got10k_train
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        got10k_val = DataLoader(
            self.got10k_val, 
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=12
        )
        return got10k_val
    
    def test_dataloader(self) -> DataLoader:
        """Return testing dataloader."""
        got10k_test = DataLoader(
            self.got10k_test, 
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=12
        )
        return got10k_test
        

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
    def __init__(self, seqs, transforms=None, max_frames_sep=50, pairs_per_seq=1):
        """Data class for generating exemplar and target images from sequences of video frames.
        
        Parameters
        ----------
        seqs : object
            Object containing list of filenames for raw images and annotations
            
        transforms : torch.Transform, default=None
            PyTorch transforms to be used for exemplar and search images
        
        max_frames_sep : int, default=50
            Maximum number of frames between exemplar and target images
        
        pairs_per_seq : int, default=1
            Number of exemplar/search pairs for each video
        """
        super().__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.max_frames_sep = max_frames_sep
        self.pairs_per_seq = pairs_per_seq
        indices = np.random.permutation(len(seqs))
        self.indices = indices[indices != 331] # We need to avoid the 332th video sequence because it's corrupted
        self.return_meta = getattr(seqs, 'return_meta', False)
    
    def __getitem__(self, index):
        # Get image filenames and annotations for video
        index = self.indices[index % len(self.indices)]
        img_files, anno  = self.seqs[index]
        frame_indices = list(range(len(img_files)))

        # Select frame indices (within maximum number of frames)
        z_idx, x_idx = self.pick_two(frame_indices)
        while x_idx - z_idx > self.max_frames_sep:
            z_idx, x_idx = self.pick_two(frame_indices)

        # Read exemplar and target images
        z = cv.imread(img_files[z_idx], cv.IMREAD_COLOR)
        x = cv.imread(img_files[x_idx], cv.IMREAD_COLOR)
        
        # Get annotations for exemplar and target images
        box_z = anno[z_idx]
        box_x = anno[x_idx]

        # Perform image transforms on exemplar and target images
        if self.transforms is not None:
            z, x = self.transforms(z, x, box_z, box_x)

        return z, x

    def __len__(self):
        return len(self.indices) *  self.pairs_per_seq

    @staticmethod
    def pick_two(indices):
        return np.sort(np.random.choice(indices, 2, replace=False))