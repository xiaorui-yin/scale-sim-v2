import math
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg

class npu_compute_fc:
    def __init__(self):
        # Params set by user
        self.config = cfg()

        self.ifmap_op_mat = np.zeros((1, 1))
        self.ofmap_op_mat = np.zeros((1, 1))
        self.filter_op_mat = np.zeros((1, 1))

        self.ifm_mm = 'broadcast'
        self.filter_mm = 'unicast'
        self.dw_flag = False

        self.filter_size = 0

        # Derived parameters
        self.Sr = 0
        self.Sc = 0

        self.num_core = 0
        self.arr_row = 0
        self.arr_col = 0
        self.num_pe = 0

        self.num_in_ch_per_step = 0
        self.num_out_ch_per_step = 0

        self.filter_fold = 0
        self.in_ch_fold = 0

        # Generated matrices
        self.ifmap_op_mat_trans = np.zeros((1,1))
        self.ifmap_prefetch_matrix = np.zeros((1,1))
        self.filter_prefetch_matrix = np.zeros((1,1))

        self.ifmap_demand_matrix = np.zeros((1,1))
        self.ofmap_demand_matrix = np.zeros((1,1))
        self.filter_demand_matrix = np.zeros((1,1))

        # Generated metrics
        self.ifmap_reads = 0
        self.filter_reads = 0
        self.ofmap_writes = 0

        self.mapping_efficiency_per_fold = []
        self.compute_utility_per_fold = []

        # Flags
        self.params_set_flag = False
        self.prefetch_mat_ready_flag = False
        self.demand_mat_ready_flag = False

        # TODO fixed local parameters, should be changed to variables in the future
        self.num_out_ch_per_group = 64

    #
    def set_params(self,
                   config_obj=cfg(),
                   ifm_mm='broadcast',
                   filter_mm='unicast',
                   filter_size=0,
                   dw_flag=False,
                   out_col=0,
                   ifmap_op_mat=np.zeros((1,1)),
                   ofmap_op_mat=np.zeros((1,1)),
                   filter_op_mat=np.zeros((1,1))
                ):

        self.config = config_obj
        self.ifmap_op_mat = ifmap_op_mat
        self.filter_op_mat = filter_op_mat
        self.ofmap_op_mat = ofmap_op_mat

        self.ifm_mm = ifm_mm
        self.filter_mm = filter_mm
        self.dw_flag = dw_flag
        self.filter_size = filter_size

        ifmap_col = self.ifmap_op_mat.shape[1]
        filter_row = self.filter_op_mat.shape[0]

        assert ifmap_col == filter_row, "Dimension mismatch between operands"

        self.Sr = self.ifmap_op_mat.shape[1]  # window size k_h * k_w * c_i
        self.Sc = self.filter_op_mat.shape[1]  # c_o
        # self.T = self.ifmap_op_mat.shape[0]  # number of output pixels

        self.num_core, self.arr_row, self.arr_col = self.config.get_array_dims()
        self.num_pe = self.num_core * self.arr_row * self.arr_col

        self.num_in_ch_per_step = self.arr_row * self.arr_col
        self.num_out_ch_per_step = self.num_core

        self.filter_fold = math.ceil(self.Sc / self.num_out_ch_per_group)
        self.in_ch_fold = math.ceil(self.Sr / self.num_in_ch_per_step)

        self.params_set_flag = True

    #
    def create_prefetch_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        self.create_ifmap_prefetch_mat()
        self.create_filter_prefetch_mat()

        self.prefetch_mat_ready_flag = True

    #
    def create_ifmap_prefetch_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.ifmap_prefetch_matrix = create_prefetch_mat(self.ifmap_demand_matrix)

    #
    def create_filter_prefetch_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.filter_prefetch_matrix = create_prefetch_mat(self.filter_demand_matrix)

    #
    def create_demand_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_demand_mat()
        self.create_filter_demand_mat()
        self.create_ofmap_demand_mat()

        assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'IFMAP and Filter demands out of sync'
        assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'OFMAP and Filter demands out of sync'
        assert self.ifmap_demand_matrix.shape[1] == self.num_pe, 'IFMAP demands exceed the rows'
        assert self.filter_demand_matrix.shape[1] == self.num_pe, 'Filter demands exceed the cols'
        assert self.ofmap_demand_matrix.shape[1] == self.num_core * self.arr_row, 'OFMAP demands exceed the cols'

        # assert len(self.compute_utility_per_fold) == self.ifmap_demand_matrix.shape[0], 'Compute utility and demand matrices out of sync'

        self.demand_mat_ready_flag = True

    # IFM demand matrix
    def create_ifmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        ifmap_demand_matrix = []

        for idx_f in range(self.filter_fold):
            this_out_ch = min((idx_f + 1) * self.num_out_ch_per_group, self.Sc) - idx_f * self.num_out_ch_per_group
            out_group_fold = math.ceil(this_out_ch / self.num_out_ch_per_step)
            for idx_ic in range(self.in_ch_fold):
                this_in_ch = min((idx_ic + 1) * self.num_in_ch_per_step, self.Sr) - idx_ic * self.num_in_ch_per_step
                ifmap_demand_row = [-1] * self.num_pe
                for paste_time in range(self.num_core):
                    for n in range(this_in_ch):
                        ifmap_demand_row[n + paste_time * self.arr_row * self.arr_col] \
                            = n + idx_ic * self.num_in_ch_per_step
                ifmap_demand_matrix.append(ifmap_demand_row)

                for idx_og in range(out_group_fold-1):
                    ifmap_demand_row = [-1] * self.num_pe
                    ifmap_demand_matrix.append(ifmap_demand_row)

        ifmap_demand_row = [-1] * self.num_pe
        ifmap_demand_matrix.append(ifmap_demand_row)

        self.ifmap_demand_matrix = np.array(ifmap_demand_matrix)

    #
    def create_filter_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        filter_demand_matrix = []

        filter_demand_row = [-1] * self.num_pe
        filter_demand_matrix.append(filter_demand_row)

        for idx_f in range(self.filter_fold):
            this_out_ch = min((idx_f + 1) * self.num_out_ch_per_group, self.Sc) - idx_f * self.num_out_ch_per_group
            out_group_fold = math.ceil(this_out_ch / self.num_out_ch_per_step)
            for idx_ic in range(self.in_ch_fold):
                this_in_ch = min((idx_ic + 1) * self.num_in_ch_per_step, self.Sr) - idx_ic * self.num_in_ch_per_step
                for idx_og in range(out_group_fold):
                    this_num_core = min((idx_og + 1) * self.num_out_ch_per_step,
                                        this_out_ch) - idx_og * self.num_out_ch_per_step

                    filter_demand_row = [-1] * self.num_pe
                    for paste_time in range(this_num_core):
                        for n in range(this_in_ch):
                            filter_demand_row[n + paste_time * self.arr_row * self.arr_col] \
                                = n + (paste_time + idx_og * self.num_out_ch_per_step + idx_f *
                                       self.num_out_ch_per_group) * self.Sr + idx_ic * self.num_in_ch_per_step
                    filter_demand_matrix.append(filter_demand_row)

        self.filter_demand_matrix = np.array(filter_demand_matrix)

    def create_ofmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        ofmap_demand_matrix = []

        num_row = self.num_core * self.arr_row

        ofmap_demand_row = [-1] * num_row
        ofmap_demand_matrix.append(ofmap_demand_row)

        for idx_f in range(self.filter_fold):
            this_out_ch = min((idx_f + 1) * self.num_out_ch_per_group, self.Sc) - idx_f * self.num_out_ch_per_group
            out_group_fold = math.ceil(this_out_ch / self.num_out_ch_per_step)
            for idx_ic in range(self.in_ch_fold):
                this_in_ch = min((idx_ic + 1) * self.num_in_ch_per_step, self.Sr) - idx_ic * self.num_in_ch_per_step
                for idx_og in range(out_group_fold):
                    this_num_core = min((idx_og + 1) * self.num_out_ch_per_step,
                                        this_out_ch) - idx_og * self.num_out_ch_per_step

                    ofmap_demand_row = [-1] * num_row
                    for paste_time in range(this_num_core):
                        ofmap_demand_row[paste_time * self.arr_row] \
                            = paste_time + idx_og * self.num_out_ch_per_step + idx_f * self.num_out_ch_per_group
                    ofmap_demand_matrix.append(ofmap_demand_row)

        self.ofmap_demand_matrix = np.array(ofmap_demand_matrix)

    # END of OFMAP demand generation

    #
    def get_ifmap_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix

    #
    def get_filter_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.filter_prefetch_matrix

    #
    def get_prefetch_matrices(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix, self.filter_prefetch_matrix

    #
    def get_ifmap_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix

    #
    def get_filter_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.filter_demand_matrix

    #
    def get_ofmap_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ofmap_demand_matrix

    #
    def get_demand_matrices(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix, self.filter_demand_matrix, self.ofmap_demand_matrix

    #
    def get_avg_mapping_efficiency(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.mapping_efficiency_per_fold)
        num = len(self.mapping_efficiency_per_fold)

        avg_mapping_eff = agg / num

        return avg_mapping_eff

    #
    def get_avg_compute_utilization(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.compute_utility_per_fold)
        num = len(self.compute_utility_per_fold)

        # avg_compute_util = agg / num

        return 0 # FIXME

    #
    def get_ifmap_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ifmap_reads

    #
    def get_filter_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.filter_reads

    #
    def get_ofmap_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ofmap_writes

    # def get_compute_mat(self):
    #     if not self.demand_mat_ready_flag:
    #         self.create_demand_matrices()
    #     
    #     return self.ifmap_compute_matrix
    #
    # def get_num_filters_per_core(self):
    #     assert self.params_set_flag, 'Parameters not set yet'
    #
    #     return self.num_filters_per_core

    def print_ofmap_trace(self, filename):
        assert self.demand_mat_ready_flag, 'Traces not generated yet'
        np.savetxt(filename, self.ofmap_demand_matrix, fmt='%d', delimiter=",")

    def print_ifmap_trace(self, filename):
        assert self.demand_mat_ready_flag, 'Traces not generated yet'
        np.savetxt(filename, self.ifmap_demand_matrix, fmt='%d', delimiter=",")

    def print_filter_trace(self, filename):
        assert self.demand_mat_ready_flag, 'Traces not generated yet'
        np.savetxt(filename, self.filter_demand_matrix, fmt='%d', delimiter=",")

#
def skew_matrix(input_matrix_np):
    rows = input_matrix_np.shape[0]
    cols = input_matrix_np.shape[1]

    out_matrix_np = np.zeros((1,1))
    for c in range(cols):
        if c == 0:
            # Comments from xyin:
            # each column represents the IFM data needed by each systolic row.
            # the first column has no data in the last row-1 clock cycles
            down_padding = -1 * np.ones((cols-1, 1))
            mat_col = input_matrix_np[:,c].reshape((rows,1))
            out_matrix_np = np.concatenate((mat_col, down_padding), axis=0)

        else:
            if c == cols -1:
                up_padding = -1 * np.ones((cols-1, 1))
                mat_col = input_matrix_np[:, c].reshape((rows, 1))

                this_col = np.concatenate((up_padding, mat_col), axis=0)
                out_matrix_np = np.concatenate((out_matrix_np, this_col), axis=1)

            else:
                # Comments from xyin:
                # other systolic rows receive data with some delays (= c),
                # and also have no data in the last row-1 clock cycles
                up_padding = -1 * np.ones((c, 1))
                mat_col = input_matrix_np[:, c].reshape((rows, 1))
                down_padding = -1 * np.ones((cols - c-1, 1))

                this_col = np.concatenate((up_padding, mat_col, down_padding), axis=0)
                out_matrix_np = np.concatenate((out_matrix_np, this_col), axis=1)

    return out_matrix_np


def create_prefetch_mat(mat):
    # for each row of mat, delete all -1 and repeated elements
    # then concatenate all rows together and reshape to 1d array

    ret = []

    for row in mat:
        tmp_row = row[row != -1]
        if len(tmp_row) != 0:
            tmp_row = np.unique(tmp_row)
            ret += tmp_row.tolist()
    return np.array(ret).reshape((1, -1))
