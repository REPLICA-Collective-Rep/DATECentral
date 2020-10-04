import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os



from .lstm import LSTMAutoencoder



# def setup_plot():
#     plt.ion()
#     fig = plt.figure()

#     gs = gridspec.GridSpec(2, 2, figure=fig)

#     inputPlot  = gs[0].subgridspec(N_FEATURES, 1)
#     inputPlots = [None] * N_FEATURES
#     for ch in range(N_FEATURES):
#         inputPlots[ch] = fig.add_subplot(inputPlot[ch,:])
#         inputPlots[ch].set_ylim(0.0, 1.0)
#         inputPlots[ch].plot(range(SEQ_LEN), np.zeros(SEQ_LEN), 'b-')

#     outputPlot  = gs[1].subgridspec(N_FEATURES, 1)
#     outputPlots = [None] * N_FEATURES
#     for ch in range(N_FEATURES):
#         outputPlots[ch] = fig.add_subplot(outputPlot[ch,:])
#         outputPlots[ch].set_ylim(0.0, 1.0)
#         outputPlots[ch].plot(range(SEQ_LEN), np.zeros(SEQ_LEN), 'r-')

#     lossAx  = fig.add_subplot(gs[2])
#     zAx     = fig.add_subplot(gs[3])



#     return fig, inputPlots, outputPlots, lossAx, zAx 


# # fig, inputPlots, outputPlots, lossAx, zAx = setup_plot()

# def plot_graph(batch, output,losses, z ):


#     for ch in range(N_FEATURES):
#         input_y = batch[:,0,ch]
#         line = inputPlots[ch].get_lines()[0]
#         line.set_ydata(input_y)

#         output_y =  output[:,0,ch]
#         line = outputPlots[ch].get_lines()[0]
#         line.set_ydata(output_y)

#     lossAx.clear()
#     lossAx.plot(losses, 'b-')


    
#     z = z[0, 0,:].squeeze()
#     zAx.clear()
#     zAx.plot(range(len(z)), z,  'b-' )

#     fig.canvas.draw()
#     plt.pause(0.0001)


class Model:

    def __init__(self, suit, seq_length, num_channels, z_dim, modeldir="models/default", writer = None):
        self.suit = suit
        self.filename = f"model_{suit}.pt"
        self.writer = writer
        self.num_channels = num_channels
        self.seq_length   = seq_length
        self.lr = 0.01

        self.model = LSTMAutoencoder(seq_length, num_channels, z_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
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

 
        if( self.n_iter % 100 == 0):
            print("Epoc: {:06d}-{:02d} Loss: {}".format(self.n_iter, self.suit, self.losses[-1]))
            self.writer.add_scalar(f"Loss/train_{self.suit}", self.losses[-1], self.n_iter)
                


            # for i in range(1):
            #     for x in range(self.seq_length):
            #         self.writer.add_histogram(f"Output/suit_{self.suit}_ch{i}", {
            #             'origional'    : batch[x,0,i],
            #             'reconstructed': output[x,0,i],
            #         }, self.n_iter )