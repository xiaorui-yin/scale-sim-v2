from torch.nn import Conv2d
import numpy as np
from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo

class compute_sim:
    def __init__(self):
        self.ifm_addr_trace = np.zeros((1, 1))
        self.ofm_addr_trace = np.zeros((1, 1))
        self.filter_addr_trace = np.zeros((1, 1))

        self.config = cfg()
        self.topo = topo()

    def set_params(self, config_obj=cfg(), topology_obj=topo()):
        self.topo = config_obj
        self.config = topology_obj
