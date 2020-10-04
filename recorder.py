import argparse
import numpy as np
import os
import time

import datastream as ds



SEQ_LEN      = 64
N_FEATURES   = 8
Z_DIM        = 32


def main(args):
    running = True

    sources = ["osc"]
    if(not args.overwrite):
        sources.append("load")

    dataserver = ds.Dataserver(SEQ_LEN , N_FEATURES, args.suits, sources, dataroot = args.dataroot)

    autosave_interval = 60 * 1

    last_save_time = time.time()
    while(running):
        try:
            dataserver.update()

            if( (time.time() - last_save_time)  >= autosave_interval):#
                dataserver.save()
                last_save_time = time.time()


        except KeyboardInterrupt:
            
            print("Clossing on interupt")
            running = False


    dataserver.save()   
    dataserver.close()

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dataroot', default="data/session_3",
                        help='Path to data')
    parser.add_argument('--overwrite',  action='store_true',
                        help='Data sources')
    parser.add_argument('--suits', nargs='+', type=int, default=[1,2,3],
                        help='Suits to train')
    args = parser.parse_args()

    main(args)