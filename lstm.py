# maple/model_lstm.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class FoliageLSTMNet(nn.Module):
    """
    Simple LSTM network for binary classification:
    predicts whether a week is a peak foliage week or not.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Probabilities with shape (batch_size,).
        """
        out, (hn, cn) = self.lstm(x)   # out: (batch, seq_len, hidden_dim)
        last_hidden = out[:, -1, :]    # take the last time step
        logits = self.fc(last_hidden)  # (batch, 1)
        probs = self.sigmoid(logits)   # (batch, 1)
        return probs.squeeze(1)        # (batch,)


@dataclass
class FoliageLSTMModel:
    """
    Wrapper around FoliageLSTMNet with training & evaluation utilities.

    This class:
    - builds sequences from weekly features
    - trains the LSTM
    - evaluates accuracy/F1/AUC
    - predicts probabilities for each week
    """
    feature_cols: Tuple[str, ...] = (
        "tavg_mean",
        "diurnal_range_mean",
        "temp_change_rate",
        "cold_day_count",
        "prcp_total",
        "dry_fraction",
    )
    seq_len: int = 4
    hidden_dim: int = 32
    num_layers: int = 1
    batch_size: int = 16
    num_epochs: int = 40
    lr: float = 1e-3
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    net: FoliageLSTMNet | None = None

    def _build_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Build sequences from a labeled weekly DataFrame.

        Returns
        -------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
        y : np.ndarray, shape (n_samples,)
        indices : list[int]
            Index of target week in original df for each sample.
        """
        df = df.reset_index(drop=True)
        features = df[list(self.feature_cols)].values.astype(np.float32)
        labels = df["is_peak"].values.astype(np.float32)

        n_weeks = len(df)
        samples_X, samples_y, indices = [], [], []

        for i in range(self.seq_len - 1, n_weeks):
            start = i - self.seq_len + 1
            end = i + 1
            seq_x = features[start:end, :]   # (seq_len, n_features)
            y = labels[i]                    # label for the current week
            samples_X.append(seq_x)
            samples_y.append(y)
            indices.append(i)

        X = np.stack(samples_X, axis=0)
        y = np.array(samples_y)
        return X, y, indices

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the LSTM on a labeled weekly DataFrame.

        Returns
        -------
        dict
            Dictionary with validation metrics: accuracy, f1, roc_auc.
        """
        if "is_peak" not in df.columns:
            raise ValueError("Training DataFrame must contain 'is_peak' column.")

        X, y, _ = self._build_sequences(df)
        n_samples = X.shape[0]
        if n_samples < 10:
            raise ValueError("Not enough weekly samples to train an LSTM model.")

        # Time-based train/test split
        train_size = int(n_samples * 0.7)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train),
                torch.from_numpy(y_train),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test),
                torch.from_numpy(y_test),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        input_dim = X.shape[2]
        self.net = FoliageLSTMNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.num_epochs):
            self.net.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                probs = self.net(batch_X)
                loss = criterion(probs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(train_loader.dataset)

        metrics = self.evaluate_on_arrays(X_test, y_test)
        print("LSTM validation metrics:", metrics)
        return metrics

    def evaluate_on_arrays(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on raw arrays (after sequence building & splitting).
        """
        if self.net is None:
            raise RuntimeError("Model has not been fitted yet.")

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_test).to(self.device)
            probs_t = self.net(X_t)
            probs = probs_t.cpu().numpy()

        y_pred = (probs >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict peak probabilities for each week in df.

        For the first (seq_len-1) weeks, there is not enough history,
        so 'peak_prob_lstm' will be NaN.

        Returns
        -------
        pandas.DataFrame
            Copy of df with an extra column 'peak_prob_lstm'.
        """
        if self.net is None:
            raise RuntimeError("Model has not been fitted yet.")

        X, _, indices = self._build_sequences(df)

        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(self.device)
            probs_t = self.net(X_t)
            probs = probs_t.cpu().numpy()

        out = df.copy()
        out["peak_prob_lstm"] = np.nan
        for idx, prob in zip(indices, probs):
            out.loc[idx, "peak_prob_lstm"] = float(prob)
        return out


if __name__ == "__main__":
    # Satisfies __name__ requirement
    print("FoliageLSTMModel is defined in this module. Use main.ipynb to run the full pipeline.")
