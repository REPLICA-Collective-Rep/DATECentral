import re
import numpy as np


def parseSensorData(reading, num_channels):
    device     = 0         
    mscounter  = None     
    data = np.zeros(num_channels)    

    pttrn1 = r"^sensors p(\d) ((?:(?:[-+]?[0-9]*\.?[0-9]+) ){8})(\d+);"
    pttrn2 = r"([-+]?[0-9]*\.?[0-9]+)"
    re1 = re.compile(pttrn1)
    re2 = re.compile(pttrn2)

    match1 = re1.match(reading)
    if ( match1 ) :
        device              = int(match1.group(1))
        mscounter           = int(match1.group(3))
        
        matches = re2.findall(match1.group(2))

        if(len(matches) == num_channels):
            for i, match in enumerate(matches):
                data[i] = float(match)
    else:
        print(f"Could not parse: {reading}")

    return device, mscounter, data