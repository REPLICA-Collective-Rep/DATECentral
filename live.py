import argparse
import zmq
import time

from dataserver import ZqmServer
from model      import Modelrunner, DumbyModel, LstmModel, ModelDef





def main(args):

    ctx = zmq.Context()
    pub_port = 5554
    sub_port = 5553
    
    if False:
        pub_ip   = "192.168.0.10"  
        sub_ip   = "0.0.0.0"    
    else:
        pub_ip   = "127.0.0.1"    
        sub_ip   = "0.0.0.0"    

    pub_addr = f"tcp://{pub_ip}:{pub_port}"
    sub_addr = f"tcp://{sub_ip}:{sub_port}"


    dataserver = ZqmServer(ctx, pub_addr, sub_addr)


    model_def = ModelDef(64, 8, 32)
    modelrunner = Modelrunner(LstmModel, model_def)


    running = True
    counter = 0
    while(running):
        try:
            sequences = dataserver.get_batch()
            if(sequences):

                print(f"----- Step {counter}")

                outputs = modelrunner.run_step( sequences )

                dataserver.set_output(outputs)

                for device, (loss, embedding) in outputs.items():
                    print(f"{device}: {loss}")


                counter += 1
                

        except KeyboardInterrupt:            
            print("Clossing on interupt")
            running = False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--new',  action='store_true',
                        help='Data sources')
    args = parser.parse_args()

    main(args)