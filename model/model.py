import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os



from .lstm import LSTMAutoencoder


class Model:

    def load(self, dir):
        raise NotImplementedError()

    def save(self, dir):
        raise NotImplementedError()

    def get_latents(self, batch):
        raise NotImplementedError()

    def train_step(self, batch):
        raise NotImplementedError()


class Model:

    def load(self, dir):
        raise NotImplementedError()

    def save(self, dir):
        raise NotImplementedError()

    def get_latents(self, batch):
        raise NotImplementedError()

    def train_step(self, batch):
        raise NotImplementedError()


class DumbyModel:

    def load(self, dir):
        pass

    def save(self, dir):
        pass

    def get_latents(self, batch):
        raise NotImplementedError()

    def train_step(self, batch):
        raise NotImplementedError()




class LstmModel(Model):

    def __init__(self, name, seq_length, num_channels, z_dim, writer = None):
        self.name = name
        self.num_channels = num_channels
        self.seq_length   = seq_length

        self.n_iter     = 0
        self.losses     = []

        self.lr = 0.01

        self.model = LSTMAutoencoder(seq_length, num_channels, z_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.loss_function = nn.SmoothL1Loss(reduction='sum')


    def load(self, dir):
        path = os.path.join(dir, self.filename)

        if(os.path.exists(path)):
            checkpoint = torch.load(path) 
            self.model.load_state_dict(checkpoint)
            self.model.eval()

            self.n_iter = 0 #### Fix this"
        else:
            print(f"No model to load {path}")


    def save(self, dir):
        path = os.path.join(dir, self.filename)
        torch.save(self.model.state_dict(), path) 

    def get_latents(self, batch):
        self.model.eval()
        batch_t = torch.from_numpy(batch.astype(np.float32)).cuda()
        output, z = self.model.forward(batch_t)

        return z.detach().cpu().numpy()

    def train_step(self, batch):
        batch_t = torch.from_numpy(batch.astype(np.float32)).cuda()

        self.model.train()
        self.optimizer.zero_grad()                   

        output, z = self.model.forward(batch_t)

        loss = self.loss_function( output, batch_t)
        loss.backward()
        self.optimizer.step()

        self.z      = z.detach().cpu().numpy()
        self.output = output.detach().cpu().numpy()

        self.n_iter += 1
        return loss.item()


                