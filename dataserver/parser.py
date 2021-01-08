import re
import numpy as np
import ctypes


Z_DIM = 32

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
        ('embedding', ctypes.c_float * Z_DIM )
    )

def parseSensorData(reading, num_channels):
    sensorData = SensorData.from_buffer_copy(reading)
    return sensorData.device, sensorData.mscounter, sensorData.raw

def encodeOutput(device, loss, embedding):
    data = embedding.astype(np.float)

    Arr = ctypes.c_float * Z_DIM
    arr = Arr()
    for i in range(Z_DIM):
        arr[i] = data[i]

    obj = OutputData(device, loss, arr)
    return obj
