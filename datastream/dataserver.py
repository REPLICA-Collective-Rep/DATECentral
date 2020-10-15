# Import needed modules from osc4py3
from osc4py3.as_comthreads import *
from osc4py3 import oscmethod as osm
from osc4py3 import oscbuildparse

import numpy as np
import time
import random
import os
import re
import datetime

import threading

HOST = "239.200.200.200"
PORT = 51555

class Datastream:

    def __init__(self, name, num_channels):
        self.itr = 0
        self.name = name
        self.num_channels = num_channels
        self.data     = np.empty((0,num_channels), np.float32)
        self.metadata = np.empty((0,1), np.int)

    def load_from_file(self, dataroot):
        path = os.path.join(dataroot, f"data_{self.name}.npy")

        if(os.path.exists(path)):
            buff     = np.load( path, self.data)
            data     = buff[:,0:self.num_channels]
            metadata = buff[:,self.num_channels:]

            if( data.shape[1] == self.num_channels):
                print(f"ch {self.name} | Loaded samples: {data.shape[0]}")
                self.data = data
                self.metadata = metadata
            else:
                print(f"ch {self.name} | Data wrong shape {data.shape}")
        else:
            print(f"ch {self.name} | Not found {path}")


    def get_sequence(self, index, seq_len):
        output = np.zeros((seq_len, self.data.shape[1]), np.float32)
        
        srt = min(index, self.data.shape[0])
        end = min(index+seq_len, self.data.shape[0])


        output[0:,:] = self.data[srt:end,:]

        return output

    def get_batch(self, seq_len, batch_size):

        if(self.data.shape[0] < seq_len):
            print(f"ch {self.name} | Not enough data... {self.data.shape[0]}")
            return None

        batch = np.empty((seq_len, 0, self.num_channels), np.float32)
        for i in range(batch_size):

            srt = random.randint(0, self.data.shape[0] - seq_len )
            end = srt + seq_len
            sample = self.data[srt:end, :]
            sample = sample.reshape((seq_len, 1, self.num_channels))

            batch = np.append(batch, sample, axis=1)

        return batch

    def save(self, dataroot):
        print(f"ch {self.name} | Saving data ({self.data.shape[0]})")
        path = f"data_{self.name}.npy"


        buff = np.append(self.data, self.metadata, axis = 1)
        np.save( os.path.join(dataroot, path), buff)



    def add_sample(self, data):
        try:
            buff          = np.array(data).reshape((1, self.num_channels + 1))
            data     =     buff[0,:self.num_channels].reshape((1, self.num_channels))
            metadata =     buff[0,self.num_channels].reshape((1,1))


            self.data     = np.append(self.data, data, axis = 0)
            self.metadata = np.append(self.metadata, metadata, axis = 0)
        except Exception as e:
            print(e)
            raise e



class Dataserver:

    def __init__(self, seq_length, num_channels, suits, sources, clients = [], dataroot="data/default"):
        self.num_channels = num_channels
        self.suits = suits
        self.sources = sources
        self.dataroot = dataroot

        self.datastreams = {}
        for suit in suits:
            self.datastreams[suit] = Datastream(suit, num_channels)

        if('osc' in self.sources):
            osc_startup()
            self.setup_clients(clients)
            self.setup_server()

        if('load' in self.sources):
            for suit in self.suits:
                self.datastreams[suit].load_from_file(dataroot)

        if('fake' in self.sources):
            self.fake_running = True
            print("Starting fake thread")
            self.fake_thread = threading.Thread(target=self.fake_loop)
            self.fake_thread.start()

    def setup_clients(self, clients):
        self.clients = []
        for idx, client in enumerate(clients):
            name = "client_{:02d}".format(idx)
            osc_udp_client(client["host"], client["port"], name)
            self.clients.append(name)

    def update(self):
        if('osc' in self.sources):
            osc_process()

    def fake_loop(self):
        while(self.fake_running):
            time.sleep(20e-3)

            t = time.time() * 10
            for suit in self.suits:
                address = "/p{}/sensor".format(suit)

                data = np.abs(np.sin(np.arange(self.num_channels) + t)) + \
                    (np.random.randn(self.num_channels) - 0.5) * 0.1
                #data = np.arange(self.num_channels)/ self.num_channels + (np.random.randn(self.num_channels) - 0.5 ) * 0.1

                self.recieve_data(address, data)
                
    def get_size(self, suit):
        return self.datastreams[suit].data.shape[0]

    def get_sequence(self, suit, index, seq_len):
        if(suit in self.datastreams):
            return self.datastreams[suit].get_sequence(index, seq_len)
        else:
            print("Suit not found")
            return None

    def get_batch(self, suit, seq_length,  batch_size, method="random"):

        if(suit in self.datastreams):
            return self.datastreams[suit].get_batch(seq_length, batch_size)
        else:
            print("Suit not found")
            return None

    def setup_server(self):
        osc_multicast_server(HOST, PORT, "bella_multicast")
        #osc_method("*", self.debug, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATA)
        osc_method("/p?/sensor", self.recieve_data,
                   argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATA)

    def send_data(self,  data):
        msg = oscbuildparse.OSCMessage(
            "/test/me", ",sif", ["text", 672, 8.871])
        for client in self.clients:
            osc_send(msg, client)

    def save(self):
        os.makedirs(self.dataroot, exist_ok = True)

        for suit in self.suits:
            self.datastreams[suit].save(self.dataroot)

    def close(self):
        if('osc' in self.sources):
            osc_terminate()

        if('fake' in self.sources):
            self.fake_running = False
            self.fake_thread.join()

    def __del__(self):
        self.close()

    def debug(self, address, data):
        print("Recieved: {}".format(address))

    def recieve_data(self, address, data):
        try:
            suit = int(address[2])
            assert(suit in self.suits)
        except Exception as e:
            print("Could not parse {}\n{}".format(address, e))
            return

        print("{} - {}".format(suit,  data))
        self.datastreams[suit].add_sample(data)
