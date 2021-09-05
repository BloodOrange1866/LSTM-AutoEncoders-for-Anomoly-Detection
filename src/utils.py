import os

from torch import nn

import copy
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING {device}')
print(torch.cuda.get_device_name(0))
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


def predict(model: object, dataset: object):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)

    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())

    return predictions, losses

def train_model(model: object, train_data: object, val_data: object, n_epochs: int) -> object:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    performance = {'train': [], 'val': []}

    optimal_model_weights = copy.deepcopy(model.state_dict())
    optimal_loss = 10000.0

    for epoch in range(1, n_epochs):
        model = model.train()

        # train step
        train_losses = []
        for each_seq in train_data:
            optimizer.zero_grad()
            X_true = each_seq.to(device)
            X_pred = model(X_true)
            loss = criterion(X_pred, X_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # valid step
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for each_seq in val_data:
                X_true = each_seq.to(device)
                X_pred = model(X_true)
                loss = criterion(X_pred, X_true)
                val_losses.append(loss.item())


        performance['train'].append(np.mean(train_losses))
        performance['val'].append(np.mean(val_losses))

        if np.mean(val_losses) < optimal_loss:
            optimal_loss = np.mean(val_losses)
            optimal_model_weights = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {np.mean(train_losses)} val loss {np.mean(val_losses)}')

    model.load_state_dict(optimal_model_weights)
    return model.eval(), performance

def save_model(model: object, save_dir: str):
    torch.save(model, save_dir)

def plot_training_performance(data: dict):
    df = pd.DataFrame([data['train'], data['val']]).T
    df.columns = ['train', 'valid']
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(df)
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction Loss")
    plt.title("LSTM Auto Encoder Reconstruction Loss")
    plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'figures', 'model.png'), dpi=400)

def plot_reconstruction_loss(data: list, title: str, fname: str, xlim: list):
    df = pd.DataFrame(data, columns=['Reconstruction Loss'])
    fig = plt.figure()
    ax = sns.distplot(df, bins=50, kde=True)
    plt.xlim(xlim[0], xlim[1])
    plt.xlabel("Reconstruction Loss Range")
    plt.title(title)
    plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'figures', fname), dpi=400)

def evaluate_model(te_loss: list, anom_loss: list, threshold: float) -> pd.DataFrame():

    def compute_acc(pred: list, threshold: float):
        predictions = []
        for p in pred:
            if p > threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    te_pred = compute_acc(pred=te_loss, threshold=threshold)
    te_true = [0] * len(te_pred)
    anom_pred = compute_acc(pred=anom_loss, threshold=threshold)
    anom_true = [1] * len(anom_pred)

    te_pred.extend(anom_pred)
    te_true.extend(anom_true)

    f1 = f1_score(te_pred, te_true, average='micro')
    return f1