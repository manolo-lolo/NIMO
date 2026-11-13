"""
PyTorch Lightning wrapper for Adaptive Ridge Regression.
"""

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def to_bin(x, n_bits):
    return (
        np.array([int(b) for b in format(x, f"0{n_bits}b")]) - 0.5
    )  # -0.5 to make it -0.5 and 0.5


class AdaptiveRidgeRegression(L.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        learning_rate=3e-4,
        lasso_penalty=0.01,
        group_penalty=1.0,
        lasso_norm=0.5,
        group_norm=0.25,
        dropout=0,
        hidden_dim=None,
        fit_intercept=False,
        noise_injection=False,
        no_feat_mask=False,
        no_pos_encoding=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        self.lambda_ = 1.0
        self.lasso_penalty = lasso_penalty
        self.group_penalty = group_penalty
        self.lasso_norm = lasso_norm
        self.group_norm = group_norm
        self.fit_intercept = fit_intercept
        self.noise_injection = noise_injection
        self.no_feat_mask = no_feat_mask
        self.no_pos_encoding = no_pos_encoding
        print(f"==> training with noise_injection={self.noise_injection}")

        self.save_hyperparameters()

        # create CO matrix
        CO = np.ones((input_dim, input_dim))  # p x p
        if not self.no_feat_mask:
            np.fill_diagonal(CO, 0.0)  # set diagonal to zero

        # create binary map, for positional encoding
        if not self.no_pos_encoding:
            self.num_pos_bits = int(np.floor(np.log2(input_dim))) + 1
            BinMap = np.vstack(
                [to_bin(i, self.num_pos_bits) for i in range(1, input_dim + 1)]
            )  # p x num_pos_bits
            CO = np.hstack([CO, BinMap])
        else:
            self.num_pos_bits = 0

        # p x (p + num_pos_bits); num_pos_bits = 0 if no_pos_encoding
        self.CO = nn.Buffer(torch.tensor(CO, dtype=torch.float32), persistent=False)

        # CO with all features zeroed, keeping only the position code; input of the baseline
        CO_zero = self.CO.clone()
        CO_zero[:, :input_dim] = 0.0
        self.CO_zero = nn.Buffer(CO_zero, persistent=False)

        # MLP definition
        if hidden_dim is not None:
            self.fc1 = nn.Linear(self.input_dim + self.num_pos_bits, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim + self.num_pos_bits, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim + hidden_dim + self.num_pos_bits, self.output_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim + self.num_pos_bits, 64)
            self.fc2 = nn.Linear(64 + self.num_pos_bits, 128)
            self.fc3 = nn.Linear(128 + 64 + self.num_pos_bits, self.output_dim)

        # self.dropout1 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.beta = nn.Buffer(torch.randn(input_dim))

        if fit_intercept:
            self.beta_0 = nn.Parameter(torch.randn(1) * 0.1)
        else:
            self.beta_0 = nn.Buffer(torch.zeros(1))

        self.c = nn.Parameter(torch.ones(self.input_dim) * 0.1)
        self.alpha2 = nn.Parameter(torch.tensor(2.0))

    def forward_MLP(self, X):
        z1 = self.fc1(X)
        # z1 = self.dropout1(z1)
        if self.training:
            if self.noise_injection:
                z1 = torch.tanh(
                    0.3 * (z1 + 0.2 * torch.randn_like(z1))
                )  # Add noise during training
            else:
                z1 = torch.tanh(0.3 * z1)
        else:
            z1 = torch.tanh(0.3 * z1)  # No noise during eval
        z1 = torch.cat([z1, X[..., self.input_dim : (self.input_dim + self.num_pos_bits)]], dim=-1)
        z2 = torch.sin(2 * np.pi * self.fc2(z1))
        z2 = self.dropout2(z2)
        z = torch.cat([z2, z1], dim=-1)
        return self.fc3(z)

    def build_B_u(self, X):
        # B_u = X + X * G_u = X * (1 + G_u); X, G_u, B_u: B x p
        return X * self.compute_nonlinearity(X)

    def forward(self, B_u):
        B_u = F.pad(B_u, (1, 0), value=1.0)  # prepend intercept column
        beta = torch.cat([self.beta_0, self.beta])

        y_hat = B_u @ beta
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        B_u = self.build_B_u(x)
        c = torch.abs(self.c)
        dc = torch.diag(c)

        X_tilde = B_u @ dc
        A = X_tilde.T @ X_tilde + self.lambda_ * torch.eye(self.input_dim, device=x.device)
        b = X_tilde.T @ y
        gamma = torch.linalg.solve(A, b)

        self.beta = c * gamma
        y_hat = self.forward(B_u)

        mse_loss = nn.MSELoss()(y, y_hat)
        lasso_loss = self.lasso_penalty * torch.sum((self.c**2) ** self.lasso_norm)

        ## Test: this loss helps to alleviate over-shrinkage
        # gamma_loss = self.lasso_penalty * torch.sum((gamma ** 2)**0.1)
        gamma_loss = 0.0
        ##

        # NOTE: the norm needs to be computed with dim=0, not dim=1
        group_loss = self.group_penalty * torch.sum(
            torch.norm(self.fc1.weight[:, : self.input_dim], dim=0) ** self.group_norm
        )

        loss = mse_loss + lasso_loss + group_loss + gamma_loss

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_mse", mse_loss, on_epoch=True, prog_bar=True)
        self.log("train_lasso", lasso_loss, on_epoch=True, prog_bar=True)
        self.log("train_group", group_loss, on_epoch=True, prog_bar=True)
        self.log("train_gamma", gamma_loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.logger.experiment.add_histogram("beta", self.beta, global_step=self.global_step)
        self.logger.experiment.add_histogram("c", self.c, global_step=self.global_step)
        self.log("beta_0", self.beta_0, on_epoch=True, prog_bar=True)
        self.log("alpha2", self.alpha2, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.MSELoss()(y, y_hat)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.MSELoss()(y, y_hat)
        print(f"Test MSE: {loss}")

    def predict_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        mse = nn.MSELoss()(y, y_hat)
        print(f"Predict MSE: {mse}")
        return y_hat

    def configure_optimizers(self):
        # beta_0 is a parameter only with fit_intercept, otherwise a buffer
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_nonlinearity(self, X):
        # X: B x p
        A_mat = F.pad(X, (0, self.num_pos_bits), value=1.0)  # B x (p + num_pos_bits)

        # one masked, position-tagged copy of each sample per feature: B x p x (p + num_pos_bits)
        inflated = rearrange(A_mat, "b d -> b 1 d") * rearrange(self.CO, "p d -> 1 p d")
        z = rearrange(self.forward_MLP(inflated), "b p 1 -> b p")

        # baseline with all features zeroed, keeping only the position code: p
        z = z - rearrange(self.forward_MLP(self.CO_zero), "p 1 -> p")

        z = torch.tanh(z)
        z = z * 0.5 * (1.0 + torch.tanh(self.alpha2))

        return z + 1  # B x p
