# lstm.py (النسخة النهائية المصححة)

import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Constructor for the inference model.
    Training-specific parameters have been removed.
    """
    def __init__(self, n_features=7, hidden_units=100, n_layers=10):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.1
        )
        self.linear = nn.Linear(self.hidden_units, 1)

    def forward(self, x):
        """
        Forward Pass - The core function needed for prediction.
        """
        # Create initial hidden and cell states on the same device as the input tensor 'x'
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units, device=x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units, device=x.device)

        # Pass the input and detached hidden/cell states through the LSTM layer
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Get the output from the last time step and pass it to the linear layer
        out = self.linear(out[:, -1, :])
        return out