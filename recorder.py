import datastream as ds
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import time


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


def main():
    running = True
    #sources = ["fake", "load"]
    sources = ["osc"]
    
    name = "test"
    dataroot = os.path.join("data", name)
    dataserver = ds.Dataserver(SEQ_LEN , N_FEATURES, SUITS, sources, dataroot = dataroot)

    autosave_interval = 60 * 2

    last_save_time = time.time()
    while(running):
        try:
            dataserver.update()

            if( (time.time() - last_save_time)  >= autosave_interval):
                dataserver.save()
                last_save_time = time.time()


        except KeyboardInterrupt:
            
            print("Clossing on interupt")
            running = False


    dataserver.save()
   
    dataserver.close()

    print("Done")

if __name__ == "__main__":
    main()