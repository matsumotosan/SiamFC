from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from pl_bolts.datasets import DummyDataset


class ImageNetDataModule(LightningDataModule):
    # TODO: change from MNIST to ImageNet
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self, stage: str = None) -> None:
        """Download and preprocess ImageNet data."""
        pass
    
    def setup(self, stage: str = None) -> None:
        """Split dataset for various stages."""
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self, batch_size: int = 32):
        """Create dataloader for training."""
        return DataLoader(self.mnist_train, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = 32):
        """Create dataloader for validation."""
        return DataLoader(self.mnist_val, batch_size=batch_size)

    def test_dataloader(self, batch_size: int = 32):
        """Create dataloader for testing."""
        return DataLoader(self.mnist_test, batch_size=batch_size)
    
    def predict_dataloader(self, batch_size: int = 32):
        """Create dataloader for inference."""
        return DataLoader(self.mnist_predict, batch_size=batch_size)