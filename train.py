import logging
import hydra
import torch
from siamfc import SiameseNet, checkpoint

# Initialize logger
log = logging.getLogger(__name__)


def _evaluate_model():
    pass

def _train_model(data_loader, model, criterion, optimizer):
    for idx, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def train(dataset, model, cfg):
    # Split data into training and validation sets    
    
    # Set optimizer
    optimizer = torch.optim.SGD(model.parameters, lr=cfg.lr, momentum=cfg.momentum)
    
    # Restore latest checkpoint if available
    model, current_epoch, stats = checkpoint.restore_checkpoint()
    
    # Evaluate model
    
    # Train model
    for epoch in range(current_epoch, cfg.n_epochs):
        _train_epoch()
        _evaluate_epoch()
        checkpoint.save_checkpoint()
    
    log.info("Finished training.")
    return None


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # Create dataset
    dataset = DataLoader()
    
    # Create model and set parameters
    embedding_net = Net()
    model = SiameseNet(embedding_net)
    
    # Train model
    train(dataset, model, cfg)


if __name__ == "__main__":
    main()
