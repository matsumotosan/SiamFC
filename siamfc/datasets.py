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


class Pair(Dataset):
    def __init__(self, seqs, transforms=None, max_frames_sep=100, pairs_per_seq=1):
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
        self.indices = np.random.permutation(len(seqs))
        self.return_meta = getattr(seqs, 'return_meta', False)

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]

        # get filename lists and annotations
        if self.return_meta:
            img_files, anno, meta = self.seqs[index]
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs[index][:2]
            vis_ratios = None
        
        # filter out noisy frames
        val_indices = self._filter(
            cv.imread(img_files[0], cv.IMREAD_COLOR),
            anno, vis_ratios)
        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        rand_z, rand_x = self._sample_pair(val_indices)

        z = cv.imread(img_files[rand_z], cv.IMREAD_COLOR)
        x = cv.imread(img_files[rand_x], cv.IMREAD_COLOR)
        z = cv.cvtColor(z, cv.COLOR_BGR2RGB)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        
        box_z = anno[rand_z]
        box_x = anno[rand_x]

        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            item = self.transforms(*item)
        
        return item
    
    def __len__(self):
        return len(self.indices) * self.pairs_per_seq
    
    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x
    
    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)
        
        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices