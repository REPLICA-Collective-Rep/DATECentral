from dataserver import FileServer
from model      import Modelrunner, ModelDef

import argparse
import time
import numpy as np
from collections import defaultdict
from pathlib import Path, PurePath
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
plt.style.use('figures/date.mplstyle')

import gc; 

list_dict = defaultdict(lambda: np.array([]))

#ffmpeg -framerate 2 -i figures/training/frames/frame_%d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p figures/training/out.mp4

DPI = 150
SIZE = (1920 / DPI, 1080 / DPI)

class SummaryPlotter:

    def __init__(self):
        self.losses = list_dict
        self.output_dir = Path("figures/training/frames4")
        self.output_dir.mkdir(parents = True, exist_ok = True)


    def plot_summary(self, epoch, outputs, eval_sequences, eval_outputs):
        width  = 6
        top    = 4
        bottom = 8

        fig = plt.figure(figsize=SIZE, dpi=DPI)
        
        gs = fig.add_gridspec(top + bottom ,width)

 
        with open("figures/training/log.txt", 'a') as f:
            f.write(f'{epoch}')
            for device, (loss, _) in outputs.items():
                f.write(f'\t{device}\t{loss}')
                self.losses[device] = np.append(self.losses[device], loss)

            loss_ax = fig.add_subplot(gs[0:top, :])
            loss_ax.autoscale(enable=True, axis='both', tight=True)

            loss_ax.xaxis.set_minor_locator(MultipleLocator(1))
            loss_ax.xaxis.set_minor_formatter(FormatStrFormatter('%d'))   
            loss_ax.xaxis.set_major_locator(MultipleLocator(50))
            loss_ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))   

            for device, losses in self.losses.items():
                loss_ax.plot(np.arange(1, len(losses) + 1), losses,  label = f"{device}")


            for c in range(width):

                for r in range(8):
                    rec_ax = fig.add_subplot(gs[top + r, c])

                    _, _, eval_output =  eval_outputs[c]
                    eval_output = eval_output.squeeze()
                    orig = eval_sequences[c].squeeze()


                    if( r == 7):

                        rec_ax.xaxis.set_major_locator(MultipleLocator(32))
                        rec_ax.xaxis.set_major_formatter(FormatStrFormatter('%d')) 
                    else:
                        rec_ax.xaxis.set_visible(False)
                        rec_ax.spines['bottom'].set_visible(False)
                        rec_ax.set_xticklabels([])

                    for _ in range(width*r*2):
                        rec_ax._get_lines.get_next_color()

                    rec_ax.plot(orig[:,r])
                    rec_ax.plot(np.arange(1, 65), eval_output[:,r])
                    rec_ax.set_ylim([0,1])

            f.write(f'{epoch}')

        fig.savefig(PurePath(self.output_dir, f"frame_{epoch}.png"))

        plt.close('all')
        gc.collect()


def main(args):

    dataserver = FileServer(
        session_dirs=[
            "data/good/2020-11-12-09-36-01-591",
            "data/good/2020-11-10-08-54-03-947"
            ]
    )

    model_def = ModelDef(64, 8, 32)
    modelrunner = Modelrunner(model_def, load_latest=True)

    eval_sequences = dataserver.get_random_batch(batch_size = 1 )
    summary = SummaryPlotter()

    batch_size = 20

    training = True
    epoch_length = 1
    epoch = 0
    while(training):

        print(f"----- Epoch {epoch}")
        for _ in range(epoch_length):
            try:
                sequences = dataserver.get_random_batch(batch_size = batch_size )
                
                
                if(sequences):
                    outputs = modelrunner.run_step( sequences )     

                for device, (loss, _) in outputs.items():
                    print(f"\t{device:04d}: {loss:0.3f}")   
                
                eval_outputs, eval_meta_outputs = modelrunner.evaluate( eval_sequences ) 

                proc=mp.Process(target=summary.plot_summary(epoch, outputs, eval_sequences, eval_outputs))
                proc.daemon=True
                proc.start()

                modelrunner.save_all()

                proc.join()
                epoch += 1

            except KeyboardInterrupt:            
                print("Clossing on interupt")
                training = False
                break
  


    modelrunner.save_all()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--new',  action='store_true',
                        help='Data sources')
    args = parser.parse_args()

    main(args)

