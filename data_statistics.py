import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, f1_score


def save_model(model, epoch, optimizer, loss, val_loss, output_folder, filename):
    path = os.path.join(output_folder, filename)

    print(f">>> Saving model to {path}")
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': loss,
                'validation_loss': val_loss
                }, path)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[:n-1] = a[:n-1]
    ret[n - 1:] = ret[n - 1:] / n
    return ret


def plot_loss(train_loss_history, val_loss_history, ax=None, n=8):
    epochs = len(train_loss_history)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(6)
        fig.set_figwidth(8)

    if (n > 0):
        ax.plot(range(1, epochs+1), moving_average(train_loss_history, n=n), label='train', linewidth=2)
        ax.plot(range(1, epochs+1), moving_average(val_loss_history, n=n), label='val', linewidth=2)
    else:
        ax.plot(range(1, epochs+1), train_loss_history, label='train', linewidth=2)
        ax.plot(range(1, epochs+1), val_loss_history, label='val', linewidth=2)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend()


def print_dataset_statistics(trainDataset, epsilon=0.1):
    num_events = len(trainDataset)
    print(f"Number of events in training dataset: {num_events}")

    num_nodes, num_edges, num_neg, num_pos = 0, 0, 0, 0
    max_nodes = -1
    for ev in trainDataset:
        num_nodes += ev.num_nodes
        num_edges += len(ev.y)
        num_pos += (ev.y > epsilon).sum()
        num_neg += (ev.y <= epsilon).sum()

        if (ev.num_nodes > max_nodes):
            max_nodes = ev.num_nodes

    print(f"Number of nodes: {num_nodes}")
    print(f"Mean Number of nodes: {num_nodes/num_events}")
    print(f"Max Number of nodes: {max_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Mean Number of edges: {num_edges/num_events}")
    print(f"Number of positive edges: {num_pos}")
    print(f"Mean Number of positive edges: {num_pos/num_events}")
    print(f"Number of negative edges: {num_neg}")
    print(f"Mean Number of negative edges: {num_neg/num_events}")


def plot_data_distribution(X, keys):
    scols = int(np.ceil(len(keys)/2))
    srows = 2
    fig, axes = plt.subplots(scols, srows, figsize=(20, 35), constrained_layout=True)

    for i, key in enumerate(keys):
        ax_col = int(i%scols)
        ax_row = int(i/scols)
        
        sns.histplot(X[key], ax=axes[ax_col, ax_row], kde=True, stat="density", linewidth=0, bins=15)
        axes[ax_col, ax_row].set_title('Frequency distribution '+ key, fontsize=18)
        axes[ax_col, ax_row].set_xlabel(key, fontsize=15)
        axes[ax_col, ax_row].set_ylabel('Count', fontsize=15)
        
    fig.tight_layout()
    plt.show()
