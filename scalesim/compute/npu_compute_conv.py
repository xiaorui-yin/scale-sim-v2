import math
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg

class npu_compute_conv:
    def __init__(self):
        # Params set by user
        self.config = cfg()

        self.ifmap_op_mat = np.zeros((1, 1))
        self.ofmap_op_mat = np.zeros((1, 1))
        self.filter_op_mat = np.zeros((1, 1))

        self.ifm_mm = 'broadcast'
        self.filter_mm = 'broadcast'
        self.dw_flag = False

        self.filter_size = 0
        self.num_in_ch = 0

        # Derived parameters
        self.Sr = 0
        self.Sc = 0
        self.T = 0

        self.out_col = 0
        self.out_row = 0
        self.window_row = 0

        self.num_core = 0
        self.arr_row = 0
        self.arr_col = 0
        self.num_pe = 0

        self.row_fold = 1
        self.col_fold = 1

        self.num_filters_per_core = 0
        self.num_rows_per_filter = 0
        self.num_filters_per_step = 0
        self.filter_fold = 0
        self.out_ch_fold = 0
        self.window_fold = 0

        self.out_row_fold = 0
        self.out_col_fold = 0

        self.num_out_px_per_step = 0
        self.out_px_fold = 0

        # Generated matrices
        self.ifmap_op_mat_trans = np.zeros((1,1))
        self.ifmap_prefetch_matrix = np.zeros((1,1))
        self.filter_prefetch_matrix = np.zeros((1,1))

        self.ifmap_demand_matrix = np.zeros((1,1))
        self.ofmap_demand_matrix = np.zeros((1,1))
        self.filter_demand_matrix = np.zeros((1,1))

        self.ifmap_compute_matrix = np.zeros((1, 1))

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

        self.num_in_ch_per_step = 0
        self.num_out_h_per_step = 0
        self.num_out_w_per_step = 0
        self.width_step = 0
        # TODO fixed local parameters, should be changed to variables in the future
        self.num_out_ch_per_group = 16

    #
    def set_params(self,
                   config_obj=cfg(),
                   ifm_mm='broadcast',
                   filter_mm='broadcast',
                   filter_size=0,
                   dw_flag=False,
                   out_col=0,
                   width_step=0,
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
        self.out_col = out_col
        self.width_step = width_step

        ifmap_col = self.ifmap_op_mat.shape[1]
        filter_row = self.filter_op_mat.shape[0]

        assert ifmap_col == filter_row, "Dimension mismatch between operands"

        self.Sr = self.ifmap_op_mat.shape[1]  # window size k_h * k_w * c_i
        self.Sc = self.filter_op_mat.shape[1]  # c_o
        self.T = self.ifmap_op_mat.shape[0]  # number of output pixels

        self.out_row = int(self.T / self.out_col)
        assert (self.T / self.out_col).is_integer(), "Must be integer"

        self.num_core, self.arr_row, self.arr_col = self.config.get_array_dims()
        self.num_pe = self.num_core * self.arr_row * self.arr_col

        # IFM unicast only supports maximum 2 OFM rows per step
        self.num_out_h_per_step = 2
        self.num_out_w_per_step = self.num_core // self.num_out_h_per_step

        self.num_in_ch = int(self.Sr / self.filter_size)
        assert (self.Sr % self.filter_size == 0), "Number of input channels is not divisible by filter size"

        # for general convolution
        self.num_in_ch_per_step = min((self.arr_row * self.arr_col) // self.filter_size, self.num_in_ch)
        # for optimized point-wise convolution TODO: what if num_in_ch_per_step = 8
        if filter_size == 1:
            self.num_in_ch_per_step = 16

        self.window_row = self.filter_size * self.num_in_ch_per_step
        self.num_rows_per_filter = int(math.pow(2, math.ceil(math.log2(self.window_row / self.arr_col))))
        assert self.num_rows_per_filter <= self.arr_row, 'Single filter cannot exceed one core'  # TODO first_layer_7x7

        self.num_filters_per_core = self.arr_row // self.num_rows_per_filter
        assert (self.arr_row / self.num_rows_per_filter).is_integer(), "Must be integer"

        if self.filter_mm == 'unicast':
            self.num_filters_per_step = self.num_core * self.num_filters_per_core
        elif self.filter_mm == 'double':
            self.num_filters_per_step = 2 * self.num_filters_per_core
        elif self.filter_mm == 'broadcast':
            self.num_filters_per_step = self.num_filters_per_core
        else:
            raise ValueError("Mapping Mode must be {'unicast', 'broadcast', 'double'}")
        self.filter_fold = math.ceil(self.Sc / self.num_out_ch_per_group)
        self.out_ch_fold = math.ceil(self.num_out_ch_per_group / self.num_filters_per_step)

        if self.ifm_mm == 'unicast':
            self.num_out_px_per_step = self.num_core
        else:
            raise ValueError("Only support unicast IFM mapping mode currently, should be extended to broadcast and double")

        self.window_fold = math.ceil(self.Sr / self.window_row)
        self.out_row_fold = math.ceil(self.out_row / self.num_out_h_per_step)
        self.out_col_fold = math.ceil(self.out_col / (self.num_out_w_per_step * self.width_step))
        self.out_px_fold = math.ceil(self.out_row_fold * self.out_col_fold)
        assert self.dw_flag is False, "Not a depthwise convolution"

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

        # all IFM elements in one OFM row are only loaded once (reuse)
        ret = []
        out_h_px_cnt = 0
        tmp = []

        for row in self.ifmap_demand_matrix:
            tmp_row = row[row != -1]
            if len(tmp_row) != 0:
                if out_h_px_cnt == 0:
                    tmp = tmp_row
                else:
                    tmp = np.concatenate((tmp, tmp_row), axis=0)
                out_h_px_cnt += self.num_out_w_per_step
                if out_h_px_cnt == self.width_step:  # FIXME, sometimes less than width_step
                    _, idx = np.unique(tmp, return_index=True)
                    tmp = tmp[np.sort(idx)]
                    ret += tmp.tolist()
                    tmp = []
                    out_h_px_cnt = 0

        self.ifmap_prefetch_matrix = np.array(ret).reshape((1, -1))

    #
    def create_filter_prefetch_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.filter_prefetch_matrix = []

        for row in self.filter_demand_matrix:
            tmp_row = row[row != -1]
            if len(tmp_row) != 0:
                _, idx = np.unique(tmp_row, return_index=True)
                tmp_row = tmp_row[np.sort(idx)]
                self.filter_prefetch_matrix += tmp_row.tolist()
        self.filter_prefetch_matrix = np.array(self.filter_prefetch_matrix).reshape((1, -1))

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

        assert len(self.compute_utility_per_fold) == self.ifmap_demand_matrix.shape[0], 'Compute utility and demand matrices out of sync'

        self.demand_mat_ready_flag = True

    # IFM demand matrix
    def create_ifmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        # clock cycles for weights to be loaded (one clock cycle)
        inter_fold_gap_prefix_mat = (np.ones((1, self.num_pe)) * -1).tolist()[0]
        # one clock cycle for the first set of filter weights to be loaded
        ifmap_compute_matrix = [inter_fold_gap_prefix_mat]
        # set is faster than list
        ifmap_compute_set = [{-1}]
        self.compute_utility_per_fold.append(0)

        for i in range(self.filter_fold):
            for j in range(self.out_px_fold):
                num_out_ch_to_process = min(self.num_out_ch_per_group, self.Sc - i * self.num_out_ch_per_group)
                for _ in range(math.ceil(num_out_ch_to_process / self.num_filters_per_step)):
                    for k in range(self.window_fold):
                        for step in range(self.width_step):
                            # ==============================================================
                            # Prepare IFM data
                            # ==============================================================
                            start_row_idx = (j // self.out_col_fold) * (2 * self.out_col) + \
                                            (j % self.out_col_fold) * self.width_step * 2 + step * 2
                            max_row_idx = (j // self.out_col_fold) * (2 * self.out_col) + self.out_col
                            if start_row_idx >= max_row_idx:
                                break
                            start_col_idx = k * self.num_in_ch_per_step

                            ifm_to_process = []

                            # process two rows at a time
                            for px_v in range(self.num_out_h_per_step):
                                for px_h in range(self.num_out_w_per_step):
                                    row_idx = start_row_idx + self.out_col * px_v + px_h

                                    # if the row index is out of bound, skip
                                    max_row_idx = (j // self.out_col_fold) * (2 * self.out_col) + self.out_col * px_v + self.out_col
                                    if row_idx >= max_row_idx:
                                        break

                                    ifm_row = []
                                    # for one output pixel, the window size is filter_size * num_in_ch_per_step
                                    for p in range(self.filter_size):
                                        max_col_idx = p * self.num_in_ch + self.num_in_ch
                                        end_col_idx = min(start_col_idx + p * self.num_in_ch + self.num_in_ch_per_step, max_col_idx)

                                        this_ = self.ifmap_op_mat[row_idx, start_col_idx + p * self.num_in_ch: end_col_idx].tolist()
                                        ifm_row += this_
                                    ifm_to_process.append(ifm_row)

                            this_fold_demand = (np.ones((self.num_core, self.arr_row * self.arr_col)) * -1).tolist()
                            num_rows_to_process = len(ifm_to_process)
                            num_cols_to_process = len(ifm_to_process[0])

                            for row in ifm_to_process:
                                assert len(row) == num_cols_to_process, 'IFM demand matrix is not a rectangle'
                                # count how many IFM elements are read (excluding padding)
                                self.ifmap_reads += len(row) - row.count(-1)

                            # ==============================================================
                            # Fill MAC cores
                            # ==============================================================
                            mac_used = 0
                            for c in range(self.num_core):
                                for f in range(self.num_filters_per_core):
                                    if f >= num_out_ch_to_process:
                                        break
                                    start_idx = f * self.num_rows_per_filter * self.arr_col
                                    end_idx = start_idx + num_cols_to_process

                                    window_idx = c * self.num_filters_per_core + f

                                    if self.ifm_mm == 'unicast':
                                        if c < num_rows_to_process:
                                            this_ifm = ifm_to_process[int(window_idx / self.num_filters_per_core)]
                                            # count how many MACs are used
                                            mac_used += num_cols_to_process
                                            # Unlike other convolutions, inside each core, IFM mode is broadcast
                                            this_fold_demand[c][start_idx: end_idx] = this_ifm
                                    else:
                                        raise ValueError("Currently only support unicast IFM mapping mode")
                            this_fold_demand = [j for sub in this_fold_demand for j in sub]

                            # calculate the overall utilization of the current compute cycle
                            this_mac_util = mac_used / self.num_pe

                            ifmap_compute_matrix.append(this_fold_demand)
                            this_set = {s for s in this_fold_demand}
                            ifmap_compute_set.append(this_set)
                            self.compute_utility_per_fold.append(this_mac_util)

        # ==============================================================
        # Create demand matrix
        # ==============================================================

        # remove reused elements for each cycle because our NPU can ensure that reused elements stay in buffer
        ifmap_demand_matrix = ifmap_compute_matrix.copy()
        for row_idx in range(len(ifmap_compute_matrix)):
            if row_idx == 0:
                continue
            for idx in range(len(ifmap_compute_matrix[0])):
                if ifmap_compute_matrix[row_idx][idx] == -1:
                    continue
                if ifmap_compute_matrix[row_idx][idx] in ifmap_compute_set[row_idx - 1]:
                    ifmap_demand_matrix[row_idx][idx] = int(-1)

        self.ifmap_compute_matrix = np.array(ifmap_compute_matrix)
        self.ifmap_demand_matrix = np.array(ifmap_demand_matrix)

    #
    def create_filter_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        filter_demand_matrix = []

        for idx_f in range(self.filter_fold):
            this_out_ch = min((idx_f + 1) * self.num_out_ch_per_group, self.Sc) - idx_f * self.num_out_ch_per_group
            for idx_oh in range(self.out_row_fold):
                this_out_h = min((idx_oh + 1) * self.num_out_h_per_step,
                                 self.out_row) - idx_oh * self.num_out_h_per_step
                for idx_ow in range(self.out_col_fold):
                    this_out_w_per_group = min((idx_ow + 1) * (self.width_step * self.num_out_w_per_step),
                                               self.out_col) - idx_ow * (self.width_step * self.num_out_w_per_step)
                    out_ch_fold = math.ceil(this_out_ch / self.num_filters_per_step)
                    for idx_oc in range(out_ch_fold):
                        this_num_filters = min((idx_oc + 1) * self.num_filters_per_step,
                                              this_out_ch) - idx_oc * self.num_filters_per_step
                        for idx_wd in range(self.window_fold):
                            this_window_row = min((idx_wd + 1) * self.window_row, self.Sr) - idx_wd * self.window_row

                            filter_demand_row = [-1] * self.num_pe
                            for paste_time in range(self.num_out_px_per_step):
                                for opx in range(this_num_filters):
                                    for wpx in range(this_window_row):
                                        filter_demand_row[wpx + (opx + self.num_filters_per_step * paste_time) *
                                                          self.num_rows_per_filter * self.arr_col] \
                                            = wpx + (opx + idx_oc * self.num_filters_per_step + idx_f *
                                                     self.num_out_ch_per_group) * self.Sr + idx_wd * self.window_row
                            filter_demand_matrix.append(filter_demand_row)

                            out_w_step_fold = math.ceil(this_out_w_per_group / self.num_out_w_per_step)
                            for idx_ws in range(out_w_step_fold-1):
                                this_out_w = min((idx_ws + 1) * self.num_out_w_per_step,
                                                 this_out_w_per_group) - idx_ws * self.num_out_w_per_step
                                filter_demand_row = [-1] * self.num_pe
                                filter_demand_matrix.append(filter_demand_row)
        filter_demand_row = [-1] * self.num_pe
        filter_demand_matrix.append(filter_demand_row)

        self.filter_demand_matrix = np.array(filter_demand_matrix)

    def create_ofmap_demand_mat(self):

        ofmap_demand_matrix = []

        num_row = self.num_core * self.arr_row
        opx_stride = num_row // (self.num_filters_per_step * self.num_out_px_per_step)

        ofmap_demand_row = [-1] * num_row
        ofmap_demand_matrix.append(ofmap_demand_row)

        for idx_f in range(self.filter_fold):
            this_out_ch = min((idx_f + 1) * self.num_out_ch_per_group, self.Sc) - idx_f * self.num_out_ch_per_group
            for idx_oh in range(self.out_row_fold):
                this_out_h = min((idx_oh + 1) * self.num_out_h_per_step,
                                 self.out_row) - idx_oh * self.num_out_h_per_step
                for idx_ow in range(self.out_col_fold):
                    this_out_w_per_group = min((idx_ow + 1) * (self.width_step * self.num_out_w_per_step),
                                               self.out_col) - idx_ow * (self.width_step * self.num_out_w_per_step)
                    out_ch_fold = math.ceil(this_out_ch / self.num_filters_per_step)
                    for idx_oc in range(out_ch_fold):
                        this_num_filters = min((idx_oc + 1) * self.num_filters_per_step,
                                               this_out_ch) - idx_oc * self.num_filters_per_step
                        for idx_wd in range(self.window_fold):
                            this_window_row = min((idx_wd + 1) * self.window_row, self.Sr) - idx_wd * self.window_row

                            # ofmap_demand_row = [-1] * num_row
                            # ofmap_demand_matrix.append(ofmap_demand_row)

                            out_w_step_fold = math.ceil(this_out_w_per_group / self.num_out_w_per_step)
                            for idx_ws in range(out_w_step_fold):
                                this_out_w = min((idx_ws + 1) * self.num_out_w_per_step,
                                                 this_out_w_per_group) - idx_ws * self.num_out_w_per_step
                                ofmap_demand_row = [-1] * num_row
                                for opx_h in range(this_out_h):
                                    for opx_w in range(this_out_w):
                                        for opx_c in range(this_num_filters):
                                            ofmap_demand_row[(opx_c + (opx_w + opx_h * self.num_out_w_per_step) *
                                                              self.num_filters_per_step) * opx_stride] = \
                                                opx_c + (opx_w + opx_h * self.out_col) * self.Sc + \
                                                idx_ws * self.Sc * self.num_out_w_per_step + \
                                                idx_ow * self.Sc * self.num_out_w_per_step * self.width_step + \
                                                idx_oh * self.Sc * self.out_col * self.num_out_h_per_step + \
                                                idx_oc * self.num_filters_per_step + idx_f * self.num_out_ch_per_group
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

        avg_compute_util = agg / num

        return avg_compute_util

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
