import argparse
import time

from dataserver import FileServer
from model      import Modelrunner, ModelDef

def main(args):

    dataserver = FileServer()


    model_def = ModelDef(64, 8, 32)
    modelrunner = Modelrunner(model_def, load_latest=True)


    running = True
    counter = 0
    while(running):
        try:
            sequences = dataserver.get_batch()
            if(sequences):

                print(f"----- Train step {counter}")

                outputs = modelrunner.run_step( sequences )

                for device, (loss, embedding) in outputs.items():
                    print(f"{device}: {loss}")


                counter += 1
                

        except KeyboardInterrupt:            
            print("Clossing on interupt")
            running = False

    modelrunner.save_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--new',  action='store_true',
                        help='Data sources')
    args = parser.parse_args()

    main(args)

