import sys 
import numpy as np
import os

from dataserver import FileServer

from pathlib import Path, PurePath
import numpy as np
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.style.use('figures/date.mplstyle')

names = [   
    "2020-11-12-09-36-01-591",
    "2020-11-10-08-54-03-947"
]

ch_names = [
    "R Shoulder",
    "R Knee",
    "R Foot",
    "R Elbow",
    "L Elbow",
    "L Foot",
    "L Knee",
    "L Shoulder"
]

DPI = 150
SIZE = (1920 / DPI, 1080 / DPI)
WINDOW = 2000

TIME_CONVERSION = 40e-3
STEP   =  floor(( 1 / 30 ) / TIME_CONVERSION) * 20

output_dir = PurePath("figures/raw_data")

session_dirs = [ os.path.join("data/good", f) for f in names ]
dataserver = FileServer( session_dirs = session_dirs)


#ffmpeg -framerate 30 -i figures/raw_data/frames_0/frame_%d.png -r 30 -c:v libx264  -pix_fmt yuv420p figures/raw_data/out_0.mp4


for index, sequence in dataserver.data.items():

    frame_dir = Path(output_dir.joinpath(f"frames_{index}"))
    frame_dir.mkdir(parents = True, exist_ok = True)

    srt = 0
    end = srt + WINDOW
    frame_num = 0

    while( end < len(sequence)):
        
        plt.close()
        fig, axs = plt.subplots(8, 1, figsize=SIZE, dpi=DPI)

        for ch in range(8):
            try:
                data = sequence[srt:end,ch+1]
                xaxis = sequence[srt:end,0]
            except Exception as e:
                print(e)
                break
 
            axs[ch].set_ylim(-0.1, 1.1) 
            axs[ch].set_yticks([0.0, 1.0])

            axs[ch].set_xlim(xaxis[0], xaxis[-1]) 


            #axs[ch].spines['left'].set_visible(False)

            if(ch == 7):
                pass
            else:
                axs[ch].xaxis.set_visible(False)
                axs[ch].spines['bottom'].set_visible(False)
                axs[ch].set_xticklabels([])

            axs[ch].set_ylabel(ch_names[ch], rotation=0, labelpad=20) 

            for _ in range(ch):
                axs[ch]._get_lines.get_next_color()

            axs[ch].plot(
                xaxis[:WINDOW], 
                data
            )

        path = f"frame_{frame_num}.png"
        plt.savefig(PurePath(frame_dir, path))
        print(f"Saving {path}")


        srt += STEP
        end += STEP
        frame_num += 1
