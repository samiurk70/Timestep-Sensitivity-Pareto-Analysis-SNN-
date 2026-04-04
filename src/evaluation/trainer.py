"""
evaluation/trainer.py

Training loop shared by SNN and ANN autoencoders.

Key design decisions:
  - MSE reconstruction loss for both models (fair comparison)
  - SNN: functional.reset_net() called before each batch (stateful neurons)
  - ANN: standard PyTorch loop, no reset needed
  - Returns per-epoch loss list for plotting
  - Checkpointing: optionally saves best model by loss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


def train(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int = 50,
    lr: float = 1e-3,
    log_every: int = 10,
    is_snn: bool = True,
    save_path: str | None = None,
) -> list[float]:
    """
    Train an autoencoder (SNN or ANN) with MSE reconstruction loss.

    Parameters
    ----------
    model      : SNNAutoencoder or ANNAutoencoder
    loader     : DataLoader yielding (batch,) tuples of normal training data
    device     : 'cuda' or 'cpu'
    epochs     : number of training epochs
    lr         : Adam learning rate
    log_every  : print loss every N epochs (set to epochs for silent training)
    is_snn     : if True, calls functional.reset_net() before each batch
    save_path  : if provided, saves best model state_dict here (.pt file)

    Returns
    -------
    List of mean epoch losses (length = epochs)
    """
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Import reset only if needed — avoids import error if spikingjelly absent
    if is_snn:
        from spikingjelly.clock_driven import functional

    epoch_losses = []
    best_loss    = float("inf")

    for epoch in range(1, epochs + 1):
        batch_losses = []

        for (batch,) in loader:
            batch = batch.to(device)

            if is_snn:
                functional.reset_net(model)

            recon = model(batch)
            loss  = criterion(recon, batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            batch_losses.append(loss.item())

        epoch_loss = float(torch.tensor(batch_losses).mean())
        epoch_losses.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{epochs}  loss={epoch_loss:.6f}")

    return epoch_losses
