"""Create checkpoint callbacks for PyTorch Lightning Trainer."""
from pytorch_lightning.callbacks import ModelCheckpoint


def setup_checkpoints(checkpoints):
    """Returns checkpoint callbacks."""
    callbacks = []
    if "val" in checkpoints:
        callbacks.append(ModelCheckpoint(
            filename="{epoch}-{step}-{val_loss:.1f}",
            monitor="val_loss",
            mode="min",
            save_top_k=-1
        ))
    if "latest" in checkpoints:
        callbacks.append(ModelCheckpoint(
            filename="latest-{epoch}-{step}",
            monitor="step",
            mode="max",
            every_n_train_steps=500,
            save_top_k=1
        ))
    return callbacks