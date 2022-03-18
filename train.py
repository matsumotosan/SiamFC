import hydra
import torch
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from siamfc import SiameseNet

# # Initialize logger
# log = logging.getLogger(__name__)


# def _evaluate_epoch(train_loader, val_loader, model, criterion, epoch):
#     stat = []
#     for data_loader in [val_loader, train_loader]:
#         y_true, y_pred, running_loss = evaluate_loop(data_loader, model, criterion)
#         total_loss = np.sum(running_loss) / y_true.size(0)
#         total_acc = accuracy(y_true, y_pred)
#         stat += [total_acc, total_loss]


# def _train_epoch(train_loader, model, criterion, optimizer):
#     for idx, (X, y) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(X)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()


# def train(train_loader, model, cfg):
#     # Split data into training and validation sets    
    
#     # Set optimizer
#     optimizer = torch.optim.SGD(model.parameters, lr=cfg.lr, momentum=cfg.momentum)
    
#     # Restore latest checkpoint if available
#     model, current_epoch, stats = checkpoint.restore_checkpoint()
    
#     # Evaluate model
    
#     # Train model
#     for epoch in range(current_epoch, cfg.n_epochs):
#         _train_epoch()
#         _evaluate_epoch()
#         checkpoint.save_checkpoint()
    
#     log.info("Finished training.")
#     return None

AVAIL_GPUS = min(1, torch.cuda.device_count())


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    
    # Initialize model
    embedding_net = None
    siamfc_model = SiameseNet(embedding_net)
    
    # Initialize DataLoader
    train_loader = DataLoader()
    
    # Initialize a trainer
    trainer = Trainer(
        max_epochs=cfg.solver.max_epochs,
        progress_bar_refresh_rate=20
    )
    
    # Train model
    trainer.fit(siamfc_model, train_loader)


if __name__ == "__main__":
    main()
