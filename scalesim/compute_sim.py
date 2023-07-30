import torch
from torch.nn import Conv2d
import numpy as np
from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from tqdm import tqdm

class compute_sim:
    def __init__(self):
        self.ifm_addr_trace = np.zeros((1, 1))
        self.ofm_addr_trace = np.zeros((1, 1))
        self.filter_addr_trace = np.zeros((1, 1))

        self.config = cfg()
        self.topo = topo()
        self.layer_id = 0

        self.ifm = torch.zeros((1, 1))
        self.ofm = torch.zeros((1, 1))
        self.kernel = torch.zeros((1, 1))

        self.ifm_trace = np.array((1, 1))
        self.ofm_trace = np.array((1, 1))
        self.kernel_trace = np.array((1, 1))

        self.params_set_flag = False
        self.golden_set_flag = False
        self.trace_load_flag = False

    def set_params(self, config_obj=cfg(), topology_obj=topo(), layer_id=0):
        self.topo = topology_obj
        self.config = config_obj
        self.layer_id = layer_id

        self.params_set_flag = True
    
    def prepare_golden_results(self):
        filter_h, filter_w = self.topo.get_filter_size(self.layer_id)
        [pad_t, pad_r, pad_b, pad_l] = self.topo.get_layer_paddings(self.layer_id)
        stride = self.topo.get_layer_strides(self.layer_id)[0]
        in_ch = self.topo.get_layer_num_channels(self.layer_id)
        out_ch = self.topo.get_layer_num_filters(self.layer_id)
        ifm_dim = self.topo.get_layer_ifmap_dims(self.layer_id)
        if self.topo.get_layer_dw_flag(self.layer_id):
            # depthwise convolution
            group = in_ch
        else:
            group = 1

        # TODO: Deconv (dilation)
        # dilation = 0
        golden_model = Conv2d(in_ch, out_ch, (filter_h, filter_w), stride, groups=group, bias=False)

        # Randomize conv2d weight
        golden_model.weight.data = torch.randn_like(golden_model.weight.data)
        self.kernel = golden_model.weight.data

        # Apply padding
        input_tensor = torch.randn(ifm_dim)
        self.ifm = torch.nn.functional.pad(input_tensor, (pad_l, pad_r, pad_t, pad_b))

        # Generate golden OFM results
        self.ofm = golden_model(self.ifm)

        self.golden_set_flag = True

    # TODO: set a load_trace mode, trace files are pre-computed and loaded
    def load_trace(self, ifm_trace, ofm_trace, kernel_trace):
        self.ifm_trace = ifm_trace
        self.ofm_trace = ofm_trace
        self.kernel_trace = kernel_trace

        assert self.ifm_trace.shape[0] == self.ofm_trace.shape[0], "Trace length dismatch"
        assert self.ifm_trace.shape[0] == self.kernel_trace.shape[0], "Trace length dismatch"

        self.trace_load_flag = True

    def run(self):
        assert self.params_set_flag and self.trace_load_flag, "Must set parameters and trace files berfore run"

        if not self.golden_set_flag:
            self.prepare_golden_results()

        for i in tqdm(range(self.ifm_trace.shape[0]), disable=pbar_disable):
            ifm = self.ifm_trace[0, :].reshape()
