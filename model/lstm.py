import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, z_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size  = n_features,
            hidden_size = z_dim,
            num_layers  = 2
        )

        self.decoder = nn.LSTM(
            input_size  = z_dim,
            hidden_size = n_features,
            num_layers  = 2
        )


    def forward(self, x):
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        return x
