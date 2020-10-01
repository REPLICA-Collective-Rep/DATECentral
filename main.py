import numpy as np
import os


import datastream as ds
from  model import Model

from   torch.utils.tensorboard import SummaryWriter


SEQ_LEN      = 64
N_FEATURES   = 8
Z_DIM        = 32
UNDER_SAMPLE = 1

CLIENTS = [
    {
        "host" : "localhost",
        "port" : 45345
    }
]

SUITS = [ 1, 2, 3 ]

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

def main():

    name = "session_1_pt"

    dataroot  = os.path.join("data", name)
    modelroot = os.path.join("models", name)


    #sources = ["fake", "load"]
    sources = ["load"]
    dataserver = ds.Dataserver(SEQ_LEN , N_FEATURES, SUITS, sources, clients = CLIENTS, dataroot = dataroot)

    os.makedirs(modelroot, exist_ok = True) 
    writer = SummaryWriter(log_dir = modelroot)
    models = { }
    for suit in SUITS:
        models[suit] = Model(suit, SEQ_LEN , N_FEATURES, Z_DIM, modelroot, False, writer)



    running = True
    while(running):
        try:

            dataserver.update()

            for suit in SUITS:
                batch = dataserver.get_batch(suit, SEQ_LEN ,  batch_size=5)

                if(batch is not None):
                    models[suit].train_step(batch)


        except KeyboardInterrupt:
            
            print("Clossing on interupt")
            running = False

    if input("Save model? (y/n):\n") == 'y':
        for suit in SUITS:
            models[suit].save(modelroot)

    if input("Save data? (y/n):\n") == 'y':
        dataserver.save()
   
    dataserver.close()

    print("Done")

if __name__ == "__main__":
    main()