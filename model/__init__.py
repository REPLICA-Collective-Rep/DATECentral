from collections import namedtuple


from .modelrunner import Modelrunner
from .model import DumbyModel, LstmModel


ModelDef = namedtuple("ModelDef", ["sequence_length", "num_channels", "z_dim"])


