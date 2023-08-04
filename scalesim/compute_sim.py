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

        self.config = cfg
        self.topo = topo
        self.layer_id = 0

        self.ifm = torch.zeros((1, 1))
        self.ofm = torch.zeros((1, 1))
        self.kernel = torch.zeros((1, 1))

        self.ifm_trace = np.array((1, 1))
        self.ofm_trace = np.array((1, 1))
        self.kernel_trace = np.array((1, 1))

        self.out_sum_level = 0

        self.params_set_flag = False
        self.golden_set_flag = False
        self.trace_load_flag = False

    def set_params(self, config_obj=cfg, topology_obj=topo, layer_id=0):
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
        ifm_dim.insert(0, in_ch)
        if self.topo.get_layer_dw_flag(self.layer_id):
            # depthwise convolution
            group = in_ch
        else:
            group = 1

        # TODO: Deconv (dilation)
        # dilation = 0
        golden_model = Conv2d(in_ch, out_ch, (filter_h, filter_w), stride, groups=group, bias=False).eval()

        # Randomize conv2d weight
        golden_model.weight.data = torch.randn_like(golden_model.weight.data)
        self.kernel = golden_model.weight.data
        # Reshape to CoxHxWxCi
        self.kernel = torch.permute(self.kernel, (0, 2, 3, 1))

        # Apply padding
        self.ifm = torch.randn(ifm_dim)
        input_tensor = torch.nn.functional.pad(self.ifm, (pad_l, pad_r, pad_t, pad_b))
        # Reshape to HWC
        self.ifm = torch.permute(self.ifm, (1, 2, 0))

        # Generate golden OFM results
        self.ofm = golden_model(input_tensor.unsqueeze(0)).squeeze(0)
        # Reshape to HWC
        self.ofm = torch.permute(self.ofm, (1, 2, 0))

        # Reshape to one-dimensional, since address trace matrices contain flattened addresses
        self.ifm = self.ifm.reshape((-1,))
        self.ofm = self.ofm.reshape((-1,))
        self.kernel = self.kernel.reshape((-1,))

        self.golden_set_flag = True

    # TODO: set a load_trace mode, trace files are pre-computed and loaded
    def load_trace(self, ifm_trace, ofm_trace, kernel_trace, out_sum_level):
        self.ifm_trace = ifm_trace
        self.ofm_trace = ofm_trace
        self.kernel_trace = kernel_trace

        self.out_sum_level = out_sum_level

        assert self.ifm_trace.shape[0] == self.ofm_trace.shape[0], "Trace length mismatch"
        assert self.ifm_trace.shape[0] == self.kernel_trace.shape[0], "Trace length mismatch"

        self.trace_load_flag = True

    def run(self):
        assert self.params_set_flag and self.trace_load_flag, "Must set parameters and trace files berfore run"

        if not self.golden_set_flag:
            self.prepare_golden_results()

        total_cycles = self.ifm_trace.shape[0]
        num_core, num_group, num_mac = self.config.get_array_dims()
        weight_reg = torch.zeros((num_core, num_group, num_mac))
        sim_ofm = torch.zeros_like(self.ofm)
        step = num_group // self.out_sum_level

        print("======================= Compute Simulation Start ======================")

        for i in tqdm(range(total_cycles)):
            ifm_addr = self.ifm_trace[i, :].reshape((num_core, num_group, num_mac))
            ifm_addr = torch.tensor(ifm_addr, dtype=torch.int64)
            ofm_addr = self.ofm_trace[i, :].reshape((num_core, num_group))
            ofm_addr = torch.tensor(ofm_addr, dtype=torch.int64)
            kernel_addr = self.kernel_trace[i, :].reshape((num_core, num_group, num_mac))
            kernel_addr = torch.tensor(kernel_addr, dtype=torch.int64)

            # If the first IFM element is unused, no computation occurs
            if i > 0:
                # Get IFM data for this cycle (exclude -1 address)
                ifm = torch.where(ifm_addr == -1, torch.zeros_like(ifm_addr), self.ifm[ifm_addr])

                # Dot-product
                tmp_out = torch.mul(weight_reg, ifm).sum(dim=-1)

                # Group results accumulation
                out = torch.zeros_like(tmp_out)
                for c in range(num_core):
                    for l in range(self.out_sum_level):
                        out[c, l * step] = tmp_out[c, l * step: (l + 1) * step].sum(dim=-1)

                # TODO: Check immediately if results are not accumulated
                # Results accumulation
                non_m1_idx = (ofm_addr != -1).nonzero()
                non_m1_addr = ofm_addr[non_m1_idx[:, 0], non_m1_idx[:, 1]]
                sim_ofm[non_m1_addr] += out[non_m1_idx[:, 0], non_m1_idx[:, 1]]

            # Update weight (if the first weight element address is -1, no update occurs)
            if kernel_addr[0, 0, 0] != -1:
                # Get Kernel data for this cycle (exclude -1 address)
                kernel = torch.where(kernel_addr == -1, torch.zeros_like(kernel_addr), self.kernel[kernel_addr])

                weight_reg = kernel

        print("=================== Simulation Results Check Start  ===================")
        if not torch.allclose(sim_ofm, self.ofm, atol=1e-3):
            print("ERROR: results mismatch")
            mismatch = torch.nonzero(sim_ofm != self.ofm)
            print("Mismatch address: ")
            print(mismatch)
        else:
            print("SUCCESS")

        print("======================= Compute Simulation End  =======================")


if __name__ == "__main__":
    sim = compute_sim()
    cfg = cfg()
    topo = topo()
    sim.set_params(cfg, topo)
    sim.load_trace(np.ones((55, 576)), np.ones((55, 64)),np.ones((55, 576)), 16)
    sim.run()

