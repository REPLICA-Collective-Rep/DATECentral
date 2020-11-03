import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os

from pathlib import Path

from .lstm import LSTMAutoencoder, Decoder


class Model:
    name = "base"
    def __init__(self, device,  model_def):
        self.device = device

        self.sequence_length  = model_def.sequence_length 
        self.num_channels     = model_def.num_channels
        self.z_dim            = model_def.z_dim

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def train_step(self, batch, live = True):
        raise NotImplementedError()


class DumbyModel(Model):
    name = "dumby"
    def __init__(self, device, model_def):
        super().__init__(device,  model_def )

    def load(self):
        pass

    def save(self):
        pass

    def train_step(self, batch, live = True):
        return (0.0, np.zeros(self.z_dim))


class MotherDecoder(Decoder):

    def __init__(self, path):

        pass

class LstmModel(Model):
    name = "lstm"
    def __init__(self, device, path, model_def):
        super().__init__(device,  model_def )
        self.path = Path(path, ".pt")

        self.losses     = []

        self.motherDecoder = MotherDecoder(path, model_def)

        self.encoder       = LSTMAutoencoder(self.decoder, self.motherEncoder)

        self.optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)

        self.loss_function = nn.SmoothL1Loss(reduction='sum')

    def exists(self):
        return self.path.exists()

    def load(self):

        if(self.exists()):
            checkpoint = torch.load(self.path) 
            self.model.load_state_dict(checkpoint)
            self.model.eval()

        else:
            print(f"No model to load {self.path}")


    def save(self, dir):
        path = os.path.join(dir, self.path)
        torch.save(self.model.state_dict(), path) 

    # def get_latents(self, batch):
    #     self.model.eval()
    #     batch_t = torch.from_numpy(batch.astype(np.float32)).cuda()
    #     output, z = self.model.forward(batch_t)

    #     return z.detach().cpu().numpy()

    def train_step(self, batch, live = True):
        batch_t = torch.from_numpy(batch.astype(np.float32)).cuda()

        if(live):
            batch_t = batch_t.reshape((self.sequence_length, 1,  self.num_channels))

        def train_step(model, batch_t, optimizer, loss_function):        
            model.train()
            optimizer.zero_grad()                   

            output, z = model.forward(batch_t)

            loss = loss_function( output, batch_t)
            loss.backward()
            optimizer.step()

            return loss, z

        loss, z = train_step(self.encoder, batch_t, self.optimizer, self.loss_function)
        self.z      = z.detach().cpu().numpy()

        if(live):
            z = z.reshape((self.z_dim))
            return loss.item(), z.detach().cpu().numpy()
        else:
            return loss.item(), None


                
