import cv2 as cv
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional
from got10k.datasets import GOT10k
from .transforms import SiamFCTransforms


class GOT10kDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule class for GOT-10k dataset."""
    def __init__(self, data_dir='./data/GOT-10k', batch_size=8) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """Download and prepare data."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Define transforms and data splits."""
        # Training and validation data
        if stage == "fit" or stage is None:
            train_seqs = GOT10k(root_dir=self.data_dir, subset='train')
            val_seqs = GOT10k(root_dir=self.data_dir, subset='val')
            self.got10k_train = Pair(seqs=train_seqs, transforms=SiamFCTransforms)
            self.got10k_val = Pair(seqs=val_seqs, transforms=SiamFCTransforms)

        # Test data
        if stage == "test" or stage is None:
            test_seqs = GOT10k(root_dir=self.data_dir, subset='test')
            self.got10k_test = Pair(seqs=test_seqs, transforms=SiamFCTransforms)

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        got10k_train = DataLoader(self.got10k_train, batch_size=self.batch_size)
        return got10k_train

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        got10k_val = DataLoader(self.got10k_val, batch_size=self.batch_size)
        return got10k_val

    def test_dataloader(self) -> DataLoader:
        """Return testing dataloader"""
        got10k_test = DataLoader(self.got10k_test, batch_size=self.batch_size)
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
    def __init__(self, seqs, transforms=None, max_frames_sep=100, pairs_per_seq=1):
        """Data class for generating exemplar and target images from sequences of video frames.

        Parameters
        ----------
        seqs : object
            Object containing list of filenames for raw images and annotations

        transforms : torch.Transform, default=None
            PyTorch transforms to be used for exemplar and target images

        max_frames_sep : int, default=100
            Maximum number of frames between exemplar and target images

        pairs_per_seq : int, default=1
            Number of target/exemplar pairs to be generated from each sequence of video frames
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

        # Select frame indices (ensure within maximum separation of number of frames)
        rand_z, rand_x = np.sort(np.random.choice(frame_indices, 2, replace=False))
        while rand_x - rand_z > self.max_frames_sep: 
            rand_z, rand_x = np.sort(np.random.choice(frame_indices, 2, replace=False))# The two chosen frames should be at most T frames apart 

        # Read exemplar and target images
        z = cv.imread(img_files[rand_z], cv.IMREAD_COLOR) #May need to be converted to RGB color space
        x = cv.imread(img_files[rand_x], cv.IMREAD_COLOR) #May need to be converted to RGB color space

        # Get annotations for exemplar and target images
        box_z = anno[rand_z]
        box_x = anno[rand_x]

        # Perform image transforms on exemplar and target images
        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            z, x = self.transforms(*item)

        return z, x

    def __len__(self):
        return len(self.indices) *  self.pairs_per_seq