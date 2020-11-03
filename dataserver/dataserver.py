import zmq
import threading
from queue import Queue
import numpy as np

from .parser import parseSensorData, encodeOutput

class DataQueueDict(dict):

    def __init__(self, maxsize):
        self.maxsize = maxsize * 2.0
        super().__init__({})

    def __missing__(self, key):
        res = self[key] = Queue(maxsize = self.maxsize)
        return res


class DataServer():
    
    def __init__(self, sequence_length, num_channels):
        self.sequence_length = sequence_length
        self.num_channels    = num_channels

    def get_batch(self):
        raise NotImplementedError()

    def set_output(self, batch):
        raise NotImplementedError()


class FileServer(DataServer):
    pass



class ZqmServer(DataServer):

    def __init__(self, ctx, pub_addr, sub_addr, sequence_length = 64, num_channels = 8):
        super().__init__(sequence_length, num_channels)
        self.ctx = ctx
        

        self.buffers = DataQueueDict(maxsize=sequence_length)


        print(f"Connecting to: {pub_addr}")
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")  # Note.
        self.sub.connect(pub_addr)

        print(f"Binding to: {sub_addr}")
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(sub_addr)


        self.thread = threading.Thread(target=self.receive_loop)
        self.running = True
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

            message = self.sub.recv()
            device, _, data = parseSensorData(message, self.num_channels )
            self._rcv_lock.acquire()
            self.buffers[device].put(data)
            self._get_lock.release()


    def set_output(self, output):
        for device, (loss, embedding) in output.items():

            msg = encodeOutput( device, loss, embedding)
            self.pub.send(msg)