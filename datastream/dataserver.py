# Import needed modules from osc4py3
from osc4py3.as_eventloop import *
from osc4py3 import oscmethod as osm
from osc4py3 import oscbuildparse



HOST = "localhost"
PORT = 51555

class Dataserver:

    def __init__(self, num_channels, clients):
        osc_startup()
        self.setup_clients(clients)
        self.setup_server()


    def setup_clients(self, clients):
        self.clients = []
        for idx, client in enumerate(clients):
            name = "client_{:02d}".format(idx)
            osc_udp_client(client["host"], client["port"], name)
            self.clients.append(name)

    def setup_server(self):
        osc_udp_server(HOST, PORT, "bella_multicast")
        osc_method("/test/*", self.receive, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATAUNPACK)


    def send_data(self, data):
        msg = oscbuildparse.OSCMessage("/test/me", ",sif", ["text", 672, 8.871])
        for client in self.clients:
            osc_send(msg, client)

    def close(self):
        osc_terminate()
        
    def __del__(self):
        self.close()

    def receive(self, address, s, x, y):
        # Will receive message address, and message data flattened in s, x, y
        pass
