import argparse
import zmq
import time

from dataserver import ZqmServer
from model      import Modelrunner

SEQ_LEN      = 64
N_FEATURES   = 8
Z_DIM        = 32


def main(args):

    # dataroot  = args.dataroot
    # modelroot = args.modelroot
    # suits     = args.suits

    ctx = zmq.Context()

    #xpub_ip   = "inanna.local"
    #xpub_ip   = "0.0.0.0"
    #pub_ip   = "192.168.8.118"
    pub_ip   = "127.0.0.1"    
    pub_port = 5554
    pub_addr = f"tcp://{pub_ip}:{pub_port}"

    sub_ip   = "127.0.0.1"    
    sub_port = 5553
    sub_addr = f"tcp://{sub_ip}:{sub_port}"


    dataserver = ZqmServer(ctx, pub_addr, sub_addr)

    modelrunner = Modelrunner(None, Z_DIM)
    while True:
        sequences = dataserver.get_batch()
        if(sequences):
            outputs = modelrunner.run_step( sequences )

            dataserver.set_output(outputs)

    # os.makedirs(modelroot, exist_ok = True) 
    # writer = SummaryWriter(log_dir = modelroot)
    # models = { }
    # for suit in suits:
    #     models[suit] = Model(suit, SEQ_LEN , N_FEATURES, Z_DIM, modelroot, writer)

    #     if(args.load):
    #         models[suit].load(modelroot)




    # running = True
    # while(running):
    #     try:

    #         dataserver.update()

    #         for suit in suits:
    #             batch = dataserver.get_batch(suit, SEQ_LEN ,  batch_size=50)

    #             if(batch is not None):
    #                 models[suit].train_step(batch)


    #     except KeyboardInterrupt:
            
    #         print("Clossing on interupt")
    #         running = False

    # if input("Save models? (y/n):\n") == 'y':
    #     for suit in suits:
    #         models[suit].save(modelroot)

    # # if input("Save data? (y/n):\n") == 'y':
    # #     dataserver.save()
   
    # dataserver.close()

    # print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    # parser.add_argument('--dataroot', default="data/session_2",
    #                     help='Path to data')
    # parser.add_argument('--modelroot', default="data/session_2",
    #                     help='Path to model')
    # parser.add_argument('--sources', nargs='+', default=["load"],
    #                     help='Data sources')
    # parser.add_argument('--load',  action='store_true',
    #                     help='Data sources')
    args = parser.parse_args()

    main(args)