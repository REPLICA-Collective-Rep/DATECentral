import torch
import torch.nn as nn
import torch.nn.functional as F


# class LSTMAutoencoder(nn.Module):

#     def __init__(self, seq_len, n_features, z_dim, num_layers = 2):
#         super(LSTMAutoencoder, self).__init__()

#         self.seq_len    = seq_len
#         self.n_features = n_features
#         self.z_dim      = z_dim
#         self.num_layers = num_layers

#         self.encoder0 = nn.LSTM(
#             input_s1ize  = n_features,
#             hidden_size  = n_features,
#             num_layers   = 1
#         )

#         self.encoder1 = nn.LSTM(
#             input_s1ize  = n_features,
#             hidden_size  = z_dim,
#             num_layers   = 1,
#             dropout      = 0.8
#         )


#     def forward(self, x):
#         c0 = torch.randn(self.num_layers, x.shape[1], self.z_dim)
#         h0 = torch.randn(self.num_layers, x.shape[1], self.z_dim)

#         z, (h1, c1) = self.encoder(x, (h0, c0))

#         z = z[-1:,:,:]
#         z_repeat = z.expand((self.seq_len,-1,-1))



#         c1 = torch.randn(self.num_layers, x.shape[1], self.n_features)
#         h1 = torch.randn(self.num_layers, x.shape[1], self.n_features)
#         x, (h2, c2) = self.decoder(z_repeat, (h1, c1))

#         print(h2.shape)

#         return x, z

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).cuda()
        self.decoder = Decoder(seq_len, n_features, embedding_dim).cuda()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, z



class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim, layers = 2):
    super(Encoder, self).__init__()

    self.seq_len    = seq_len
    self.n_features = n_features
    self.layers     = layers

    self.embedding_dim = embedding_dim
    self.hidden_dim    = 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size  = n_features,
      hidden_size = self.hidden_dim,
      num_layers  = layers,
      dropout      = 0.8
    )

    self.rnn2 = nn.LSTM(
      input_size  = self.hidden_dim,
      hidden_size = embedding_dim,
      num_layers  = 1,
    )

    self.rnn3 = nn.LSTM(
      input_size  = self.hidden_dim,
      hidden_size = embedding_dim,
      num_layers  = 1,
    )

  def forward(self, x):
    c0 = torch.randn(self.layers, x.shape[1], self.hidden_dim).cuda()
    h0 = torch.randn(self.layers, x.shape[1], self.hidden_dim).cuda()

    x, (_, _) = self.rnn1(x, (c0, h0))


    c1 = torch.randn(1, x.shape[1], self.embedding_dim).cuda()
    h1 = torch.randn(1, x.shape[1], self.embedding_dim).cuda()

    _, (mu, _)     = self.rnn2(x, (c1, h1))
    _, (logvar, _) = self.rnn3(x, (c1, h1))

    return mu, logvar


class Decoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim, layers = 2):

    super(Decoder, self).__init__()

    self.seq_len    = seq_len
    self.n_features = n_features
    self.layers     = layers

    self.embedding_dim = embedding_dim
    self.hidden_dim    = 2 * embedding_dim


    self.rnn1 = nn.LSTM(
      input_size   = embedding_dim,
      hidden_size  = embedding_dim,
      num_layers   = layers,
      dropout      = 0.8
    )

    self.rnn2 = nn.LSTM(
      input_size   = embedding_dim,
      hidden_size  = self.hidden_dim,
      num_layers   = 1 
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat((self.seq_len,1,1))

    c0 = torch.randn(self.layers, x.shape[1], self.embedding_dim).cuda()
    h0 = torch.randn(self.layers, x.shape[1], self.embedding_dim).cuda()

    x, (hidden_n, cell_n) = self.rnn1(x, (c0, h0))

    c1 = torch.randn(1, x.shape[1], self.hidden_dim).cuda()
    h1 = torch.randn(1, x.shape[1], self.hidden_dim).cuda()

    x, (hidden_n, cell_n) = self.rnn2(x, (c1, h1))

    return self.output_layer(x)
