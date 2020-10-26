import argparse
import numpy as np
import os

import datastream as ds
from  model import Model

from torch.utils.tensorboard import SummaryWriter


SEQ_LEN      = 64
N_FEATURES   = 8
Z_DIM        = 32


def main(args):

    dataroot  = args.dataroot
    modelroot = args.modelroot
    suits     = args.suits
    sources   = args.sources

    dataserver = ds.Dataserver(SEQ_LEN , N_FEATURES, suits, sources, clients = CLIENTS, dataroot = dataroot)

    os.makedirs(modelroot, exist_ok = True) 
    writer = SummaryWriter(log_dir = modelroot)
    models = { }
    for suit in suits:
        models[suit] = Model(suit, SEQ_LEN , N_FEATURES, Z_DIM, modelroot, writer)

        if(args.load):
            models[suit].load(modelroot)




    running = True
    while(running):
        try:

            dataserver.update()

            for suit in suits:
                batch = dataserver.get_batch(suit, SEQ_LEN ,  batch_size=50)

                if(batch is not None):
                    models[suit].train_step(batch)


        except KeyboardInterrupt:
            
            print("Clossing on interupt")
            running = False

    if input("Save models? (y/n):\n") == 'y':
        for suit in suits:
            models[suit].save(modelroot)

    # if input("Save data? (y/n):\n") == 'y':
    #     dataserver.save()
   
    dataserver.close()

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dataroot', default="data/session_2",
                        help='Path to data')
    parser.add_argument('--modelroot', default="data/session_2",
                        help='Path to model')
    parser.add_argument('--sources', nargs='+', default=["load"],
                        help='Data sources')
    parser.add_argument('--load',  action='store_true',
                        help='Data sources')
    parser.add_argument('--suits', nargs='+', type=int, default=[1,2,3],
                        help='Suits to train')
    args = parser.parse_args()

    main(args)