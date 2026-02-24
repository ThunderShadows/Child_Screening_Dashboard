"""
Temporal Deep Learning Model (LSTM) for CECI Framework.
Processes ordered session sequences to estimate PID and PEff.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import os
import json


class CECITemporalModel(nn.Module):
    """
    LSTM model with two output heads:
        - PID: Probability of Persistent Cognitive Difficulty
        - PEff: Probability of Low/Inconsistent Effort

    Architecture:
        Input (batch, seq_len, n_features)
          → LSTM encoder (shared)
          → FC head 1 → PID (sigmoid)
          → FC head 2 → PEff (sigmoid)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # PID head (persistent cognitive difficulty)
        self.pid_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # PEff head (low/inconsistent effort)
        self.peff_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, n_features)
            lengths: (batch,) actual sequence lengths

        Returns:
            pid: (batch, 1) probability of persistent cognitive difficulty
            peff: (batch, 1) probability of low effort
        """
        # Pack sequences if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
            )
            lstm_out, (h_n, _) = self.lstm(packed)
        else:
            lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from top layer
        h_last = h_n[-1]  # (batch, hidden_size)

        pid = self.pid_head(h_last)
        peff = self.peff_head(h_last)

        return pid, peff


class TemporalModelTrainer:
    """Handles training, evaluation, saving, and loading of the LSTM model."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 0.001,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CECITemporalModel(
            input_size, hidden_size, num_layers, dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
        }

    def _prepare_labels(
        self, labels: np.ndarray, var_acc: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create training labels for both heads.

        PID labels: 1 if child is at_risk (label=1), 0 otherwise
        PEff labels: derived from accuracy variance — high variance suggests effort issues
        """
        pid_labels = torch.FloatTensor(labels).unsqueeze(1).to(self.device)

        # PEff: high variance + not at_risk → likely effort issue
        # Normalize variance to [0,1] range
        var_norm = (var_acc - var_acc.min()) / (var_acc.max() - var_acc.min() + 1e-9)
        # Effort issues are more likely when: high variance AND not at risk
        peff_labels = torch.FloatTensor(
            var_norm * (1 - labels) * 0.8 + var_norm * labels * 0.2
        ).unsqueeze(1).to(self.device)

        return pid_labels, peff_labels

    def train(
        self,
        X_seq: np.ndarray,
        labels: np.ndarray,
        var_acc: np.ndarray,
        seq_lengths: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict:
        """
        Train the temporal model.

        Args:
            X_seq: (n_children, max_seq_len, n_features) session sequences
            labels: (n_children,) binary labels
            var_acc: (n_children,) accuracy variance for PEff label generation
            seq_lengths: (n_children,) actual sequence lengths
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training metrics dictionary
        """
        pid_labels, peff_labels = self._prepare_labels(labels, var_acc)

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        len_tensor = torch.LongTensor(seq_lengths).to(self.device)

        dataset = TensorDataset(X_tensor, len_tensor, pid_labels, peff_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        pid_criterion = nn.BCELoss()
        peff_criterion = nn.BCELoss()

        history = {"loss": [], "pid_loss": [], "peff_loss": []}

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_pid_loss = 0.0
            epoch_peff_loss = 0.0
            n_batches = 0

            for batch_x, batch_len, batch_pid, batch_peff in loader:
                self.optimizer.zero_grad()

                pred_pid, pred_peff = self.model(batch_x, batch_len)
                loss_pid = pid_criterion(pred_pid, batch_pid)
                loss_peff = peff_criterion(pred_peff, batch_peff)
                loss = loss_pid + 0.5 * loss_peff  # PID is primary objective

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_pid_loss += loss_pid.item()
                epoch_peff_loss += loss_peff.item()
                n_batches += 1

            history["loss"].append(round(epoch_loss / n_batches, 4))
            history["pid_loss"].append(round(epoch_pid_loss / n_batches, 4))
            history["peff_loss"].append(round(epoch_peff_loss / n_batches, 4))

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Loss: {history['loss'][-1]:.4f} | "
                    f"PID Loss: {history['pid_loss'][-1]:.4f} | "
                    f"PEff Loss: {history['peff_loss'][-1]:.4f}"
                )

        # Final predictions on full dataset
        self.model.eval()
        with torch.no_grad():
            pred_pid, pred_peff = self.model(X_tensor, len_tensor)
            pred_pid = pred_pid.cpu().numpy().flatten()
            pred_peff = pred_peff.cpu().numpy().flatten()

        metrics = {
            "final_loss": history["loss"][-1],
            "final_pid_loss": history["pid_loss"][-1],
            "final_peff_loss": history["peff_loss"][-1],
            "pid_mean": round(float(np.mean(pred_pid)), 4),
            "peff_mean": round(float(np.mean(pred_peff)), 4),
            "training_history": history,
        }
        return metrics

    def predict(
        self, X_seq: np.ndarray, seq_lengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference. Returns (PID_probabilities, PEff_probabilities).
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            len_tensor = torch.LongTensor(seq_lengths).to(self.device)
            pid, peff = self.model(X_tensor, len_tensor)
            return pid.cpu().numpy().flatten(), peff.cpu().numpy().flatten()

    def save(self, dir_path: str):
        """Save model weights and config."""
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dir_path, "lstm_weights.pth"))
        with open(os.path.join(dir_path, "lstm_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, dir_path: str, device: str = None) -> "TemporalModelTrainer":
        """Load model from disk."""
        with open(os.path.join(dir_path, "lstm_config.json")) as f:
            config = json.load(f)
        trainer = cls(**config, device=device)
        trainer.model.load_state_dict(
            torch.load(
                os.path.join(dir_path, "lstm_weights.pth"),
                map_location=trainer.device,
                weights_only=True,
            )
        )
        trainer.model.eval()
        return trainer
