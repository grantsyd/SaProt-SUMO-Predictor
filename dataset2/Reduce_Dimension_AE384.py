# -*- coding: utf-8 -*-
"""
Reduce_Dimension_AE384.py

Reduce high-dimensional SaProt features to 384-dimensional latent features
using an autoencoder.

Input:
    features/dataset2_saprot_center_mean.npy

Output:
    features/dataset2_saprot_center_mean_AE384.npy
    trained_model/ae384_encoder.pt
    trained_model/ae384_scaler.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 384, dropout: float = 0.10):
        super().__init__()
        hidden1 = max(1024, latent_dim * 2)
        hidden2 = max(512, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X, latent_dim, epochs, batch_size, lr, device):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AutoEncoder(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 15
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            rec = model(batch)
            loss = loss_fn(rec, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                rec = model(batch)
                loss = loss_fn(rec, batch)
                val_loss += loss.item() * len(batch)
        val_loss /= val_size

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def encode_all(model, X, batch_size, device):
    model.eval()
    latents = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            z = model.encode(batch).detach().cpu().numpy()
            latents.append(z.astype(np.float32))
    return np.concatenate(latents, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="features/dataset2_saprot_center_mean.npy")
    parser.add_argument("--out_feature", default="features/dataset2_saprot_center_mean_AE384.npy")
    parser.add_argument("--model_out", default="trained_model/ae384_encoder.pt")
    parser.add_argument("--scaler_out", default="trained_model/ae384_scaler.pkl")
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Path(args.out_feature).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    X_raw = np.load(args.feature).astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    model, best_val = train_autoencoder(X, args.latent_dim, args.epochs, args.batch_size, args.lr, args.device)
    Z = encode_all(model, X, args.batch_size, args.device)

    np.save(args.out_feature, Z)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": int(X.shape[1]),
        "latent_dim": int(args.latent_dim),
        "best_val_loss": float(best_val),
    }, args.model_out)
    joblib.dump(scaler, args.scaler_out)

    print(f"Saved latent feature: {args.out_feature}, shape={Z.shape}")
    print(f"Saved AE model: {args.model_out}")
    print(f"Saved scaler: {args.scaler_out}")


if __name__ == "__main__":
    main()
