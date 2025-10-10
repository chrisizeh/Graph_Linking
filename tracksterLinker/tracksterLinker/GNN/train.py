from tqdm import tqdm
import numpy as np
import random

import torch

from tracksterLinker.GNN.LossFunctions import FocalLoss
from tracksterLinker.datasets.GNNDataset import GNNDataset
from tracksterLinker.utils.dataUtils import calc_weights
from tracksterLinker.utils.perturbations.inErrorBars import *

def train(model, opt, loader, epoch, weighted="raw_energy", scores=False, emb_out=False, loss_obj=FocalLoss(), node_feature_dict=GNNDataset.node_feature_dict):

    epoch_loss = 0

    model.train()
    for sample in tqdm(loader, desc=f"Training Epoch {epoch}"):

        # reset optimizer and enable training mode
        opt.zero_grad()
        emb, z = model.run(sample.x, sample.edge_features, sample.edge_index)
        weights = calc_weights(sample.edge_index, sample.x, node_feature_dict, name=weighted)
        # print("z", z)

        # compute the loss
        if scores:
            label = torch.full((sample.edge_index.shape[0], ), random.getrandbits(1), device=sample.x.device)
            if label[0] == 0:
                dupl = perturbate(sample.x, num_samples=1, with_z=True)
                emb_dupl, _ = model.run(dupl.squeeze(0), sample.edge_features, sample.edge_index)
            else:
                indices = torch.randperm(sample.edge_index.shape[0])
                emb_dupl = emb[indices]

            loss = loss_obj(z.squeeze(-1), emb.squeeze(-1), emb_dupl.squeeze(-1), sample.y, label, weights)
        else:
            # print(z[z > 1])
            # print(sample.y[sample.y > 1])
            # print(weights[weights > 1])
            loss = loss_obj(z.squeeze(-1), torch.ceil(sample.y), weights)

        # back-propagate and update the weight
        loss.backward()
        opt.step()
        epoch_loss += loss

    return float(epoch_loss)/len(loader)


def test(model, loader, epoch, weighted="raw_energy", scores=False, loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), node_feature_dict=GNNDataset.node_feature_dict):

    with torch.set_grad_enabled(False):
        model.eval()
        val_loss = 0.0

        # 0: tp, 1: fp, 2: fn, 3: tn
        cross_edges = torch.zeros(4, device=device)
        signal_edges = torch.zeros(4, device=device)
        pu_edges = torch.zeros(4, device=device)
            
        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            nn_emb, nn_pred = model.run(sample.x, sample.edge_features, sample.edge_index)
            weights = calc_weights(sample.edge_index, sample.x, node_feature_dict, name=weighted)

            y_pred = (nn_pred > model.threshold).squeeze()
            y_true = (sample.y > 0).squeeze()

            cross_edges[0] += torch.sum(weights[sample.PU_info[:, 0]] * (y_true[sample.PU_info[:, 0]] & y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[1] += torch.sum(weights[sample.PU_info[:, 0]] * (~y_true[sample.PU_info[:, 0]] & y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[2] += torch.sum(weights[sample.PU_info[:, 0]] * (y_true[sample.PU_info[:, 0]] & ~y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[3] += torch.sum(weights[sample.PU_info[:, 0]] * (~y_true[sample.PU_info[:, 0]] & ~y_pred[sample.PU_info[:, 0]])).item()

            signal_edges[0] += torch.sum(weights[sample.PU_info[:, 1]] * (y_true[sample.PU_info[:, 1]] & y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[1] += torch.sum(weights[sample.PU_info[:, 1]] * (~y_true[sample.PU_info[:, 1]] & y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[2] += torch.sum(weights[sample.PU_info[:, 1]] * (y_true[sample.PU_info[:, 1]] & ~y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[3] += torch.sum(weights[sample.PU_info[:, 1]] * (~y_true[sample.PU_info[:, 1]] & ~y_pred[sample.PU_info[:, 1]])).item()
            
            pu_edges[0] += torch.sum(weights[sample.PU_info[:, 2]] * (y_true[sample.PU_info[:, 2]] & y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[1] += torch.sum(weights[sample.PU_info[:, 2]] * (~y_true[sample.PU_info[:, 2]] & y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[2] += torch.sum(weights[sample.PU_info[:, 2]] * (y_true[sample.PU_info[:, 2]] & ~y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[3] += torch.sum(weights[sample.PU_info[:, 2]] * (~y_true[sample.PU_info[:, 2]] & ~y_pred[sample.PU_info[:, 2]])).item()
            
            if scores:
                val_loss += loss_obj(nn_pred.squeeze(-1), nn_emb.squeeze(-1), sample.y, sample.PU_info, weights).item()
            else:
                # print(nn_pred)
                # print(sample.y)
                val_loss += loss_obj(nn_pred.squeeze(-1), sample.y, weights).item()

        val_loss /= len(loader)
        return val_loss, cross_edges, signal_edges, pu_edges 


def validate(model, loader, epoch, weighted="raw_energy", scores=False, loss_obj=FocalLoss(), node_feature_dict=GNNDataset.node_feature_dict):

    with torch.set_grad_enabled(False):
        model.eval()
        val_loss = 0.0

        pred, y, weights = [], [], []
        PU_info = [[], [], []]
            
        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            nn_emb, nn_pred = model.run(sample.x, sample.edge_features, sample.edge_index)
            pred += nn_pred.squeeze(-1).tolist()
            y += sample.y.tolist()
            weight = calc_weights(sample.edge_index, sample.x, node_feature_dict, name=weighted)
            weights += weight.tolist()

            PU_info[0] += sample.PU_info[:, 0].tolist()
            PU_info[1] += sample.PU_info[:, 1].tolist()
            PU_info[2] += sample.PU_info[:, 2].tolist()
            
            if scores:
                val_loss += loss_obj(nn_pred.squeeze(-1), nn_emb.squeeze(-1), sample.y, sample.PU_info, weight).item()
            else:
                val_loss += loss_obj(nn_pred.squeeze(-1), torch.ceil(sample.y), weight).item()

        val_loss /= len(loader)
    return val_loss, torch.tensor(pred), torch.tensor(y), torch.tensor(weights), torch.tensor(PU_info)
