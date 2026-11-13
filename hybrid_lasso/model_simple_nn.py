"""
Pytorch Lightning wrapper for the Simple NN model.
"""

import torch
import torch.nn as nn
import lightning as L

import numpy as np


class SimpleNN(L.LightningModule):
    def __init__(self, input_dim, output_dim, task="reg", lr=1e-3, dropout=0, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.task = task
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.save_hyperparameters()

        if hidden_dim is not None:
            self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.fc3 = nn.Linear(self.hidden_dim * 2, self.output_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim, 64)
            self.fc2 = nn.Linear(64, 128)
            self.fc3 = nn.Linear(128 + 64, self.output_dim)

        self.dropout1 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        z1 = torch.tanh(self.fc1(x))
        z1 = self.dropout1(z1)
        z2 = torch.sin(2 * np.pi * self.fc2(z1))
        z2 = self.dropout2(z2)
        z = torch.cat([z2, z1], dim=1)
        return self.fc3(z).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        y_pred = self(x)

        if self.task == "cls":
            loss = nn.BCEWithLogitsLoss()(y_pred.squeeze(), y)
            y_hat = torch.sigmoid(y_pred)
            acc = (y_hat.round() == y).float().mean()
            self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        else:
            loss = nn.MSELoss()(y_pred, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        y_pred = self(x)

        if self.task == "cls":
            loss = nn.BCEWithLogitsLoss()(y_pred.squeeze(), y)
            y_hat = torch.sigmoid(y_pred)
            acc = (y_hat.round() == y).float().mean()
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        else:
            loss = nn.MSELoss()(y_pred, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        y_pred = self(x)

        if self.task == "cls":
            loss = nn.BCEWithLogitsLoss()(y_pred.squeeze(), y)
            y_hat = torch.sigmoid(y_pred)
            acc = (y_hat.round() == y).float().mean()
            self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        else:
            loss = nn.MSELoss()(y_pred, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        y_pred = self(x)

        if self.task == "cls":
            y_hat = torch.sigmoid(y_pred)
            pred = (y_hat > 0.5).float()

            acc = (pred == y).float().mean()
            print(f"Test ACC: {acc}")
        else:
            pred = y_pred

        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
