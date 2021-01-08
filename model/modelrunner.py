
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from .model import EncoderModel, DecoderModel

class Modelrunner():

    def __init__(self, model_def, load_latest = False ):
        self.models_root = "models"
        self.model_def   = model_def

        last_session = self.getLastSession()
        if(load_latest):
            self.session = last_session
        else:
            self.session = last_session + 1

        
        self.encoders = {}
        self.metaencoder = self.setupEncoder()
        self.decoder = self.setupDecoder()

        self.loss_function = nn.SmoothL1Loss(reduction='sum')
        # self.fig, self.inputPlots, self.outputPlots = self.setup_plot()
    
    def save_all(self):
        print("Saving all models...")
        for encoder in self.encoders.values():
            encoder.save()

        self.metaencoder.save()
        self.decoder.save()
        print("All models saves")


    def run_step(self, sequences):
        # if( self.n_iter % 50 == 0):
        #     print("Epoc: {:06d}-{:02d} Loss: {}".format(self.n_iter, self.name, self.losses[-1]))
        #     self.writer.add_scalar(f"Loss/train_{self.name}", self.losses[-1], self.n_iter)
        #     self.plot_graph(batch, self.output)
        #     self.writer.add_figure(f"Samples/train_{self.name}", self.fig, self.n_iter)
        outputs = {}

        decoder = self.decoder
        metaencoder = self.metaencoder

        decoder.model.train()
        metaencoder.model.train()

        decoder.zero_grad()
        metaencoder.zero_grad()
        for device, sequence in sequences.items():

            batch_t = torch.from_numpy(sequence.astype(np.float32)).cuda()

            if(batch_t.ndim == 2):
                batch_t = batch_t.reshape(
                    (self.model_def.sequence_length, 1,  self.model_def.num_channels)
                )

            if( device != 0):
                if( device not in self.encoders): 
                    self.encoders[device] = self.setupEncoder(f"encoder_{device:02d}")    

                encoder = self.encoders[device]
                encoder.model.train()

                # Encoder
                encoder.zero_grad() 
                loss, z = self.run_model(encoder, decoder, batch_t)
                loss.backward()
                encoder.optimizer.step()

                # Metaencoder
                meta_loss, _ = self.run_model(metaencoder, decoder, batch_t)
                meta_loss.backward()

                z = z.squeeze().detach().cpu().numpy()
                loss = loss.item()
                outputs[device] = (loss, z)

            else:
                # Metaencoder
                meta_loss, meta_z = self.run_model(metaencoder, decoder, batch_t)
                meta_loss.backward()

                loss = meta_loss.item()
                outputs[device] = (loss, None)              

        metaencoder.optimizer.step()
        decoder.optimizer.step()

        return outputs

    def run_model(self, encoder, decoder, batch_t):
        mu, logvar =  encoder.model(batch_t)
        z = self.reparameterize(mu, logvar)

        output = decoder.model(z)

        loss = self.loss_function(output, batch_t)

        return loss, z

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
        
    def setupEncoder(self, name = None):

        if(name):
            path = Path(self.models_root, f"Session_{self.session:04d}", name)
        else:
            path = Path(self.models_root, "metaencoder")

        model = EncoderModel(path, self.model_def)
        model.load()

        return model

    def setupDecoder(self):
        path = Path(self.models_root, "decoder")
        model = DecoderModel(path, self.model_def)
        model.load()

        return model

    def getLastSession(self):
        path = Path(self.models_root)

        sessions = []
        for p in path.glob('Session_*'):
            if(p.is_dir()):
                try:
                    sessions.append(int(p.stem[-4:]))
                except:
                    pass
        if(len(sessions)):
            return max(sessions)
        else:
            return 0


    # def setup_plot(self):
    #     fig = plt.figure()

    #     gs = gridspec.GridSpec(1, 2, figure=fig)

    #     inputPlot  = gs[0].subgridspec(self.num_channels, 1)
    #     inputPlots = [0.5] * self.num_channels
    #     for ch in range(self.num_channels):
    #         inputPlots[ch] = fig.add_subplot(inputPlot[ch,:])
    #         inputPlots[ch].set_ylim(0.0, 1.0)
    #         inputPlots[ch].plot(range(self.seq_length ), np.zeros(self.seq_length ), 'b-')

    #     outputPlot  = gs[1].subgridspec(self.num_channels, 1)
    #     outputPlots = [0.5] * self.num_channels
    #     for ch in range(self.num_channels):
    #         outputPlots[ch] = fig.add_subplot(outputPlot[ch,:])
    #         outputPlots[ch].set_ylim(0.0, 1.0)
    #         outputPlots[ch].plot(range(self.seq_length ), np.zeros(self.seq_length ), 'r-')

    #     # zAx     = fig.add_subplot(gs[3])
        
    #     return fig, inputPlots, outputPlots

    # def plot_graph(self, batch, output ):

    #     for ch in range(self.num_channels):
    #         input_y = batch[:,0,ch]
    #         line = self.inputPlots[ch].get_lines()[0]
    #         line.set_ydata(input_y)

    #         output_y =  output[:,0,ch]
    #         line = self.outputPlots[ch].get_lines()[0]
    #         line.set_ydata(output_y)