import os.path as osp
import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.pyplot as plt

from tracksterLinker.datasets.GNNDataset import GNNDataset
from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet, FocalLoss, EarlyStopping, weight_init
from tracksterLinker.GNN.train import *
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.graphUtils import print_graph_statistics
from tracksterLinker.utils.plotResults import *


load_weights = False
model_name = "model_2025-07-04_epoch_27_epoch_27_dict.pt"

base_folder = "/home/czeh"
model_folder = osp.join(base_folder, "GNN/model")
hist_folder = osp.join(base_folder, "histo_CloseByMP_0PU")
data_folder_training = osp.join(base_folder, "GNN/dataset")
data_folder_test = osp.join(base_folder, "GNN/dataset_test")
os.makedirs(model_folder, exist_ok=True)

# Prepare Dataset
batch_size = 1
dataset_test = GNNDataset(data_folder_test, hist_folder, test=True)
dataset_training = GNNDataset(data_folder_training, hist_folder)
train_dl = DataLoader(dataset_training, shuffle=True, batch_size=batch_size)
test_dl = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Prepare Model
epochs = 200

model = GNN_TrackLinkingNet(input_dim=len(dataset_training.model_feature_keys),
                            edge_feature_dim=dataset_training.get(0).edge_features.shape[1],
                            edge_hidden_dim=16, hidden_dim=16, weighted_aggr=True, dropout=0.3,
                            node_scaler=dataset_training.node_scaler, edge_scaler=dataset_training.edge_scaler)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
loss_obj = FocalLoss(alpha=0.45, gamma=2)
early_stopping = EarlyStopping(patience=20, delta=-2)

model.apply(weight_init)

# Load weights if needed
start_epoch = 0

if load_weights:
    weights = torch.load(osp.join(model_folder, model_name), weights_only=True)
    model.load_state_dict(weights["model_state_dict"])
    optimizer.load_state_dict(weights["optimizer_state_dict"])
    start_epoch = weights["epoch"]

train_loss_hist = []
val_loss_hist = []
date = f"{datetime.now():%Y-%m-%d}"

save_model(model, 0, optimizer, [], [], output_folder=model_folder, filename=f"model_empty", dummy_input=dataset_training.get(0))
for epoch in range(start_epoch, epochs):
    print(f'Epoch: {epoch+1}')

    loss = train(model, optimizer, train_dl, epoch+1, device=device, loss_obj=loss_obj)
    train_loss_hist.append(loss)

    val_loss, pred, y, weight = test(model, test_dl, epoch+1, loss_obj=loss_obj, device=device, weighted="raw_energy")
    val_loss_hist.append(val_loss)
    print(f'Training loss: {loss}, Validation loss: {val_loss}')

    plot_loss(train_loss_hist, val_loss_hist, save=True, output_folder=model_folder, filename=f"model_date_{date}_loss_epochs")
    plot_validation_results(pred, y, save=True, output_folder=model_folder, file_suffix=f"epoch_{epoch+1}_date_{date}", weight=weight)
    save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"model_{date}", dummy_input=dataset_training.get(0))
    early_stopping(model, val_loss)

    if early_stopping.early_stop:
        print(f"Early stopping after {epoch+1} epochs")
        early_stopping.load_best_model(model)

        save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"model_{date}_final_loss_{-early_stopping.best_score:.4f}", dummy_input=dataset_training.get(0))
        break

    scheduler.step()
    plt.close()
