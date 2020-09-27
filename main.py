import datastream as ds
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import LSTMAutoencoder

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



SEQ_LEN      = 32
N_FEATURES   = 8
Z_DIM        = 16
UNDER_SAMPLE = 1

CLIENTS = [
    {
        "host" : "localhost",
        "port" : 45345
    }
]

#SUITS = [ 1, 2, 3 ]
SUITS = [ 1 ]

def setup_plot():
    plt.ion()
    fig = plt.figure()

    gs = gridspec.GridSpec(2, 2, figure=fig)

    inputPlot  = gs[0].subgridspec(N_FEATURES, 1)
    inputPlots = [None] * N_FEATURES
    for ch in range(N_FEATURES):
        inputPlots[ch] = fig.add_subplot(inputPlot[ch,:])
        inputPlots[ch].set_ylim(0.0, 1.0)
        inputPlots[ch].plot(range(SEQ_LEN), np.zeros(SEQ_LEN), 'b-')

    outputPlot  = gs[1].subgridspec(N_FEATURES, 1)
    outputPlots = [None] * N_FEATURES
    for ch in range(N_FEATURES):
        outputPlots[ch] = fig.add_subplot(outputPlot[ch,:])
        outputPlots[ch].set_ylim(0.0, 1.0)
        outputPlots[ch].plot(range(SEQ_LEN), np.zeros(SEQ_LEN), 'r-')

    lossAx  = fig.add_subplot(gs[2])
    zAx     = fig.add_subplot(gs[3])



    return fig, inputPlots, outputPlots, lossAx, zAx 


fig, inputPlots, outputPlots, lossAx, zAx = setup_plot()

def plot_graph(batch, output,losses, z ):


    for ch in range(N_FEATURES):
        input_y = batch[:,0,ch]
        line = inputPlots[ch].get_lines()[0]
        line.set_ydata(input_y)

        output_y =  output[:,0,ch]
        line = outputPlots[ch].get_lines()[0]
        line.set_ydata(output_y)

    lossAx.clear()
    lossAx.plot(losses, 'b-')


    
    z = z[0, 0,:].squeeze()
    zAx.clear()
    zAx.plot(range(len(z)), z,  'b-' )

    fig.canvas.draw()
    plt.pause(0.0001)

def main():
    running = True
    dataserver = ds.Dataserver(SEQ_LEN , N_FEATURES, SUITS, CLIENTS, fake = False)

    
    #loss_function = nn.MSELoss(reduction='sum')
    loss_function = nn.SmoothL1Loss(reduction='sum')

    #loss_function = nn.L1Loss(reduction='mean')
    #optimizer = optim.SGD(model.parameters(), lr=0.01)   
    
    optimizers = {}
    models = {}
    epoch  = {}
    losses = {}
    for suit in SUITS:
        models[suit] = LSTMAutoencoder(SEQ_LEN, N_FEATURES, Z_DIM)
        epoch[suit]  = 0
        losses[suit] = []
        optimizers[suit] = optim.Adam(models[suit].parameters(), lr=1e-3)


    while(running):
        try:

            dataserver.update()

            for suit in SUITS:
                batch = dataserver.get_batch(suit=suit, batch_size=5)

                if(batch is not None):
                    models[suit].train()

                    batchT = torch.from_numpy(batch.astype(np.float32)).cuda()


                    optimizers[suit].zero_grad()                   

                    output, z = models[suit].forward(batchT)
                    loss = loss_function( output, batchT)

                    losses[suit].append(loss.item())

                    
                    loss.backward()
                    optimizers[suit].step()

                    z      = z.detach().cpu().numpy()
                    output = output.detach().cpu().numpy()


                    if(suit == 1):
                        plot_graph(batch, output, losses[suit], z )

                    print("Epoc: {:04d}-{:02d} Loss: {}".format(epoch[suit], suit, loss))
                    epoch[suit] += 1

                

        except KeyboardInterrupt:
            
            print("Clossing on interupt")
            running = False


    dataserver.close()
    fig.clf()
    print("Done")

if __name__ == "__main__":
    main()