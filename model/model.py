import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os

from pathlib import Path

from .lstm import Encoder, Decoder


class Model:
    def __init__(self, path, model_def):
        self.sequence_length  = model_def.sequence_length 
        self.num_channels     = model_def.num_channels
        self.z_dim            = model_def.z_dim

        self.path = Path(path).with_suffix( ".pt")

    
    def zero_grad(self):
        if(self.optimizer):
            self.optimizer.zero_grad()

    def exists(self):
        return self.path.exists()

    def load(self):

        if(self.exists()):
            checkpoint = torch.load(self.path) 
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"Loaded {self.path}")
        else:
            print(f"No model to load {self.path}")

    def save(self):
        if(self.model):
            parent_dir = self.path.parents[0]
            if(not parent_dir.exists()):
                parent_dir.mkdir(parents=True, exist_ok=False)


            torch.save(self.model.state_dict(), self.path) 

    def train_step(self, batch, live = True):
        raise NotImplementedError()


class EncoderModel(Model):
    def __init__(self, path, model_def):
        super().__init__( path, model_def )

        self.model = Encoder(
            self.sequence_length,
            self.num_channels,   
            self.z_dim          
        )
        self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
               

class DecoderModel(Model):
    def __init__(self, path, model_def):
        super().__init__( path, model_def )

        self.model = Decoder(
            self.sequence_length,
            self.num_channels,   
            self.z_dim          
        )
        self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

                
