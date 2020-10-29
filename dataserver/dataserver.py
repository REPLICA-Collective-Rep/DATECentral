import zmq
import threading
from queue import Queue
import numpy as np

from .parser import parseSensorData

class DataServer():
    

    def __init__(self, sequence_length, num_channels):
        self.sequence_length = sequence_length
        self.num_channels    = num_channels


class FileServer(DataServer):
    pass


class DataQueue(dict):

    def __init__(self, maxsize):
        self.maxsize = maxsize * 2.0
        super().__init__({})

    def __missing__(self, key):
        res = self[key] = Queue(maxsize = self.maxsize)
        return res

class ZqmServer(DataServer):


    def __init__(self, ctx, xpub_addr, sequence_length = 64, num_channels = 8):
        super().__init__(sequence_length, num_channels)
        self.ctx = ctx
        

        self.buffers = DataQueue(sequence_length)

        self.running = True

        print(f"Connecting to: {xpub_addr}")
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Note.
        self.socket.connect(xpub_addr)

        self.thread = threading.Thread(target=self.receive_loop)
        self.thread.start()

        self._rcv_lock = threading.Lock()
        self._get_lock = threading.Lock()
        self._get_lock.acquire()



    def get_batch(self):#
        sequences = {}

        self._get_lock.acquire()
        for key, buffer in self.buffers.items():

            if( buffer.qsize() > self.sequence_length):
                sequence = []

                
                for i in range(self.sequence_length):
                    sequence.append(buffer.get())
                

                sequences[key] = np.asarray(sequence)
                
        self._rcv_lock.release()
        return sequences


    def receive_loop(self):
        while(self.running):

            message = self.socket.recv()
            device, _, data = parseSensorData(message.decode("utf-8"), self.num_channels )

            self._rcv_lock.acquire()
            self.buffers[device].put(data)
            self._get_lock.release()
