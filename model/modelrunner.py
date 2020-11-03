
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path

class Modelrunner():

    def __init__(self, model_type, z_dim, load_latest = True ):
        self.models_root = "models"
        self.z_dim = z_dim
        self.load_latest = load_latest
        self.model_type = model_type


        self.models = {}
        self.encoder = None
        self.decoder = None

        last_session = self.getLastSession()
        if(self.load_latest):
            self.session = last_session
        else:
            self.session = last_session + 1

        # self.fig, self.inputPlots, self.outputPlots = self.setup_plot()

    def run_step(self, sequences):
        # if( self.n_iter % 50 == 0):
        #     print("Epoc: {:06d}-{:02d} Loss: {}".format(self.n_iter, self.name, self.losses[-1]))
        #     self.writer.add_scalar(f"Loss/train_{self.name}", self.losses[-1], self.n_iter)
        #     self.plot_graph(batch, self.output)
        #     self.writer.add_figure(f"Samples/train_{self.name}", self.fig, self.n_iter)
        outputs = {}

        
        for device, sequence in sequences.items():
            if( device not in self.models):
                self.setupModel(device)                

            outputs[device] =  self.models[device].train_step(sequence)


        return outputs

    def setupModel(self, device):

        path = Path(self.models_root, f"Session_{self.session:04d}", f"encoder_{device:02d}")
        self.models[device] = self.model_type(device, path, self.z_dim)

        if(self.load_latest and self.models[device].exists()):
            self.models[device].load()

    def getLastSession(self):
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