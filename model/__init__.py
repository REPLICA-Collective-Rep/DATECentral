from collections import namedtuple


from .modelrunner import Modelrunner


ModelDef = namedtuple("ModelDef", ["sequence_length", "num_channels", "z_dim"])


