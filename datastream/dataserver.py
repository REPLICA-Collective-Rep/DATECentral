# Import needed modules from osc4py3
from osc4py3.as_comthreads import *
from osc4py3 import oscmethod as osm
from osc4py3 import oscbuildparse

import numpy as np
import time
import random
import os
import re

import threading

HOST = "239.200.200.200"
PORT = 51555

MAX_BUFFER = 100

class Datastream:

    def __init__(self, name, seq_length, num_channels, undersample):
        self.itr          = 0
        self.name         = name
        self.seq_length   = seq_length
        self.num_channels = num_channels
        self.undersample  = int(undersample)
        self.buffer_size  = seq_length * num_channels
        self.newSequences = []
        self.sequences    = []


    def load_from_file(self, path):
        print(path)
        with open(path, "r") as f:
            d = f.read()



        expr1 = re.compile(r"(?:\[(.*?)\])")
        matches = expr1.finditer(d)
        sequences = []
        for i, match in enumerate(matches):
            
            seq_str = match.group(1)

            expr2 = re.compile(r"(?:(\d\.\d*)(?:, )?)")
            matches2 = expr2.finditer(seq_str)
            
            sequence = tuple( float(m.group(1)) for m in matches2)
            sequences.append(sequence)
            print("{} -> {}".format(i, len(sequence)))


        self.sequences = np.asarray(sequences)

    def get_batch(self, batch_size):
        if(len(self.sequences) + len(self.newSequences) < (batch_size + 1)):
            return None

        batch = self.newSequences[0:batch_size]

        if(batch != [] ):
            if( len(batch[-1]) < self.buffer_size):
                batch = self.newSequences[0:len(batch) - 1]

        paddingNum = batch_size - len(batch)
        padding = random.choices(self.sequences, k = paddingNum)
        
        if(any(batch)):
            self.sequences.extend(batch) 
            self.newSequences = self.newSequences[len(batch):-1] 
        batch.extend(padding)

        batch = np.asarray(batch, dtype=object)
        batch = np.reshape( batch, (batch_size, self.seq_length, self.num_channels))
        batch = np.swapaxes(batch, 0, 1)

        #print("Get {} | New: {} Old: {}".format(self.name, len(self.newSequences), len(self.sequences)))

        return batch


    def save(self):
        file = "data/suit_{}.dump".format(self.name)
        with open(file,  "a+") as f:
            for s in self.sequences:
                f.write("{}".format(s))

        self.sequences = []


    def add_sample(self, data):
        self.itr += 1

        # if(self.itr % self.undersample != 0):
        #     return

        assert(len(data) == self.num_channels)

        if(len(self.newSequences) > 0):
            if(len(self.newSequences[-1]) < self.buffer_size):
                self.newSequences[-1].extend(list(data))
            else:
                self.newSequences.append(list(data))
        else:
            self.newSequences.append(list(data))

        if(len(self.newSequences) > MAX_BUFFER):
            self.newSequences.extend(self.newSequences[0])
            self.newSequences.remove(0)
            print("Buffer overflow")


        if(len(self.sequences) > 10000):
            self.save()

        #print("Add {} | New: {} Old: {}".format(self.name, len(self.newSequences), len(self.sequences)))

        




class Dataserver:

    def __init__(self, seq_length, num_channels, suits, clients, undersample = 1, fake = False):
        self.num_channels = num_channels
        self.fake = fake        
        self.suits = suits

        if(not fake):
            osc_startup()
            self.setup_clients(clients)
            self.setup_server()
        else:
            self.fake_running = True
            self.fake_thread =  threading.Thread(target=self.fake_loop)
            self.fake_thread.start()

        self.datastreams = {}
        for suit in suits:
            self.datastreams[suit] = Datastream(suit, seq_length, num_channels, undersample)
    
    def load(self, path):

        for suit in self.suits:

            path = os.path.join(path, f"suit_{suit}.dump")
            self.datastreams[suit].load_from_file(path)

    def setup_clients(self, clients):
        self.clients = []
        for idx, client in enumerate(clients):
            name = "client_{:02d}".format(idx)
            osc_udp_client(client["host"], client["port"], name)
            self.clients.append(name)


    def update(self):
        if not self.fake:
            osc_process()
        else:
            pass


    def fake_loop(self):
        while( self.fake_running ):
            time.sleep( 20e-3)

            t = time.time() * 10
            for suit in self.suits:
                address = "/p{}/sensor".format(suit)

                data = np.abs(np.sin(np.arange(self.num_channels) + t )) + (np.random.randn(self.num_channels) - 0.5 ) * 0.1
                #data = np.arange(self.num_channels)/ self.num_channels + (np.random.randn(self.num_channels) - 0.5 ) * 0.1

                self.recieve_data(address, data)

    def get_batch(self, suit, batch_size):

        if(suit in self.datastreams):
            return self.datastreams[suit].get_batch(batch_size)
        else:
            print("Suit not found")
            return None


    def setup_server(self):
        osc_multicast_server(HOST, PORT, "bella_multicast")
        #osc_method("*", self.debug, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATA)
        osc_method("/p?/sensor", self.recieve_data, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATA)


    def send_data(self, 
    data):
        msg = oscbuildparse.OSCMessage("/test/me", ",sif", ["text", 672, 8.871])
        for client in self.clients:
            osc_send(msg, client)

    def close(self):
        if not self.fake:
            
            for suit in self.suits:
                self.datastreams[suit].save()

            osc_terminate()
        else:
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
            print("Could not parse {}".address)
            print(e)
            return

        #print("{} - {}".format(suit,  data))
        self.datastreams[suit].add_sample(data)

