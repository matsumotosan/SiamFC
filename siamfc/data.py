from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class ImageNetDataModule(LightningDataModule):
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        return DataLoader()

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()