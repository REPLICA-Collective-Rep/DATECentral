import re
import numpy as np
import ctypes

class SensorData(ctypes.Structure):
    _fields_ = (
        ('device'   , ctypes.c_int),
        ('mscounter', ctypes.c_double),
        ('raw'      , ctypes.c_float * 8)
    )

class OutputData(ctypes.Structure):
    _fields_ = (
        ('device'   , ctypes.c_int),
        ('loss'     , ctypes.c_float),
        ('embedding', ctypes.c_float * 16)
    )

def parseSensorData(reading, num_channels):
    sensorData = SensorData.from_buffer_copy(reading)
    return sensorData.device, sensorData.mscounter, sensorData.raw

def encodeOutput(device, loss, embedding):
    data = embedding.astype(np.single)

    Arr = ctypes.c_float * 16
    arr = Arr()
    for i in range(16):
        arr[i] = data[i]

    return OutputData(device, loss, arr)