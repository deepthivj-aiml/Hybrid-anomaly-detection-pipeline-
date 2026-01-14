import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_features, hidden1=16, hidden2=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, n_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)