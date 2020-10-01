import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os



from .lstm import LSTMAutoencoder

class Model:

    def __init__(self, suit, seq_length, num_channels, z_dim, modeldir="models/default", load = False, writer = None):
        self.suit = suit
        self.filename = f"model_{suit}.pt"
        self.writer = writer

        self.model = LSTMAutoencoder(seq_length, num_channels, z_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-1)
        #optimizer = optim.SGD(model.parameters(), lr=0.01) 

        self.loss_function = nn.SmoothL1Loss(reduction='sum')
        #loss_function = nn.MSELoss(reduction='sum') 
        
        self.n_iter     = 0
        self.losses     = []


    def load(self, dir):

        path = os.path.join(dir, self.filename)

        if(os.path.exists(path)):
            checkpoint = torch.load(path) 
            self.model.load_state_dict(checkpoint)
            self.model.eval()
        else:
            print(f"No model to load {path}")


    def save(self, dir):
        path = os.path.join(dir, self.filename)
        torch.save(self.model.state_dict(), path) 

    def train_step(self, batch):
        batch = torch.from_numpy(batch.astype(np.float32)).cuda()

        self.model.train()
        self.optimizer.zero_grad()                   

        output, z = self.model.forward(batch)

        loss = self.loss_function( output, batch)

        self.losses.append(loss.item())

        loss.backward()
        self.optimizer.step()

        self.z      = z.detach().cpu().numpy()
        self.output = output.detach().cpu().numpy()

        self.n_iter += 1

        if(self.writer is not None):
            self.writer.add_scalar(f"Loss/train_{self.suit}", self.losses[-1], self.n_iter)

        if( self.n_iter % 100 == 0):
            print("Epoc: {:06d}-{:02d} Loss: {}".format(self.n_iter, self.suit, self.losses[-1]))
                



# class ModelZoo:


#     def __init__(self, suits, seq_length, num_channels, z_dim, modeldir="models/default", load = False):
#         self.modeldir = modeldir
#         self.models = [ Model(suit, seq_length, num_channels, z_dim ) for suit in suits ]



#         if(load):
#             self.load()

#     def load(self):
#        for model in self.models:
#            model.load(self.modeldir)
