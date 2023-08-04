import math
import numpy as np
import copy
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

        self.filter_size = 0
        self.num_in_ch = 0

        # Derived parameters
        self.Sr = 0
        self.Sc = 0
        self.T = 0

        self.out_col = 0
        self.out_row = 0
        self.window_part_size = 0

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
        self.filter_size = filter_size
        self.out_col = out_col

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


        self.num_in_ch = int(self.Sr / self.filter_size)
        assert (self.Sr % self.filter_size == 0), "Number of input channels is not divisible by filter size"

        # for general convolution
        # TODO: make this as input parameter or calculate the optimal value
        self.num_in_ch_per_step = min((self.arr_row * self.arr_col) // self.filter_size, self.num_in_ch)
        if filter_size == 1:
            self.num_in_ch_per_step = 16

        self.window_part_size = self.filter_size * self.num_in_ch_per_step
        self.num_rows_per_filter = int(math.pow(2, math.ceil(math.log2(self.window_part_size / self.arr_col))))
        assert self.num_rows_per_filter <= self.arr_row, 'Single filter cannot exceed one core'  # TODO first_layer_7x7

        self.num_filters_per_core = self.arr_row // self.num_rows_per_filter
        assert (self.arr_row / self.num_rows_per_filter).is_integer(), "Must be integer"

        if self.filter_mm == 'unicast':
            self.num_filters_per_step = self.num_core * self.num_filters_per_core
            if self.num_filters_per_step > self.num_out_ch_per_group:
                self.num_out_ch_per_group = self.num_filters_per_step
            self.filter_fold = math.ceil(self.Sc / self.num_out_ch_per_group)
        elif self.filter_mm == 'double':
            self.num_filters_per_step = 2 * self.num_filters_per_core
        elif self.filter_mm == 'broadcast':
            self.filter_fold = math.ceil(self.Sc / self.num_out_ch_per_group)
            self.num_filters_per_step = self.num_filters_per_core
        else:
            raise ValueError("Mapping Mode must be {'unicast', 'broadcast', 'double'}")
        # self.filter_fold = math.ceil(self.Sc / self.num_out_ch_per_group)
        self.out_ch_fold = math.ceil(self.num_out_ch_per_group / self.num_filters_per_step)

        if self.ifm_mm == 'unicast':
            self.num_out_px_per_step = self.num_core
            # TODO: if having more than four cores, need to balance horizontal and vertical OFM pixel numbers
            self.num_out_h_per_step = 2
            self.num_out_w_per_step = self.num_core // self.num_out_h_per_step
        elif self.ifm_mm == 'broadcast':
            if self.filter_mm != 'unicast':
                raise ValueError("No supported mapping mode combination")
            # 1 x 1 x num_core
            self.num_out_px_per_step = self.num_core
            self.num_out_h_per_step = 1
            self.num_out_w_per_step = 1
        else:
            raise ValueError("Only support unicast IFM mapping mode currently, should be extended to broadcast and double")

        self.width_step = width_step // self.num_out_w_per_step

        self.window_fold = math.ceil(self.Sr / self.window_part_size)
        self.out_row_fold = math.ceil(self.out_row / self.num_out_h_per_step)
        self.out_col_fold = math.ceil(self.out_col / (self.num_out_w_per_step * self.width_step))
        self.out_px_fold = math.ceil(self.out_row_fold * self.out_col_fold)

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

        self.ifmap_prefetch_matrix = self.ifmap_demand_matrix[self.ifmap_demand_matrix != -1].reshape((1, -1))
        self.ifmap_reads = self.ifmap_prefetch_matrix.shape[1]

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
        self.filter_reads = self.filter_prefetch_matrix.shape[1]

    #
    def create_demand_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_fmap_demand_mat()
        self.create_filter_demand_mat()
        # self.create_ofmap_demand_mat()

        assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'IFMAP and Filter demands out of sync'
        assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'OFMAP and Filter demands out of sync'
        assert self.ifmap_demand_matrix.shape[1] == self.num_pe, 'IFMAP demands exceed the rows'
        assert self.filter_demand_matrix.shape[1] == self.num_pe, 'Filter demands exceed the cols'
        assert self.ofmap_demand_matrix.shape[1] == self.num_core * self.arr_row, 'OFMAP demands exceed the cols'

        assert len(self.compute_utility_per_fold) == self.ifmap_demand_matrix.shape[0], 'Compute utility and demand matrices out of sync'

        self.demand_mat_ready_flag = True

    # IFM demand matrix
    def create_fmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        # clock cycles for weights to be loaded (one clock cycle)
        inter_fold_gap_prefix_ifm = (np.ones((1, self.num_pe)) * -1).tolist()[0]
        inter_fold_gap_prefix_ofm = (np.ones((1, self.num_core * self.arr_row)) * -1).tolist()[0]
        # one clock cycle for the first set of filter weights to be loaded
        ifmap_compute_matrix = [inter_fold_gap_prefix_ifm]
        ofmap_demand_matrix = [inter_fold_gap_prefix_ofm]
        # set is faster than list
        ifmap_compute_set = [{-1}]
        self.compute_utility_per_fold.append(0)

        for i in range(self.filter_fold):
            for j in range(self.out_px_fold):
                num_out_ch_to_process = min(self.num_out_ch_per_group, self.Sc - i * self.num_out_ch_per_group)
                for out_ch_step in range(math.ceil(num_out_ch_to_process / self.num_filters_per_step)):
                    for k in range(self.window_fold):
                        for step in range(self.width_step):
                            # ==============================================================
                            # Prepare IFM/OFM data
                            # ==============================================================
                            
                            # start_row_idx is the index of the first processed OFM element in this step
                            start_row_idx = (j // self.out_col_fold) * (self.num_out_h_per_step * self.out_col) + \
                                            (j % self.out_col_fold) * self.width_step * self.num_out_w_per_step + step * self.num_out_w_per_step
                            max_row_idx = (j // self.out_col_fold) * (self.num_out_h_per_step * self.out_col) + self.out_col
                            if start_row_idx >= max_row_idx:
                                break
                            start_col_idx = k * self.num_in_ch_per_step

                            ifm_to_process = []
                            ofm_to_process = []

                            # process two rows at a time
                            for px_v in range(self.num_out_h_per_step):
                                for px_h in range(self.num_out_w_per_step):
                                    row_idx = start_row_idx + self.out_col * px_v + px_h

                                    # in the case of odd number out_row or out_col
                                    if start_row_idx % self.out_col + px_h >= self.out_col or \
                                        start_row_idx // self.out_col + px_v >= self.out_row:
                                        break

                                    # if the row index is out of bound, skip
                                    max_row_idx = (j // self.out_col_fold) * (self.num_out_h_per_step * self.out_col) + self.out_col * px_v + self.out_col
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

                                    if self.filter_mm == 'unicast':
                                        for c in range(self.num_core):
                                            start_out_ch_idx = i * self.num_out_ch_per_group + \
                                                    out_ch_step * self.num_filters_per_step + \
                                                    c * self.num_filters_per_core
                                            if start_out_ch_idx >= self.Sc:
                                                break
                                            end_out_ch_idx = min(start_out_ch_idx + self.num_filters_per_core, self.Sc)
                                            ofm_to_process.append(self.ofmap_op_mat[row_idx, start_out_ch_idx: end_out_ch_idx])
                                    elif self.filter_mm == 'broadcast':
                                        start_out_ch_idx = i * self.num_out_ch_per_group + out_ch_step * self.num_filters_per_step
                                        if start_out_ch_idx >= self.Sc:
                                            break
                                        end_out_ch_idx = min(start_out_ch_idx + self.num_filters_per_step, self.Sc)
                                        ofm_to_process.append(self.ofmap_op_mat[row_idx, start_out_ch_idx: end_out_ch_idx])

                            num_rows_to_process = len(ifm_to_process)
                            num_cols_to_process = len(ifm_to_process[0])

                            for row in ifm_to_process:
                                assert len(row) == num_cols_to_process, 'IFM demand matrix is not a rectangle'

                            # ==============================================================
                            # Fill MAC cores
                            # ==============================================================

                            this_ifm_demand = (np.ones((self.num_core, self.arr_row * self.arr_col)) * -1).tolist()
                            this_ofm_demand = (np.ones((self.num_core, self.arr_row)) * -1).tolist()
                            mac_used = 0
                            for c in range(self.num_core):
                                for f in range(self.num_filters_per_core):
                                    start_idx = f * self.num_rows_per_filter * self.arr_col
                                    end_idx = start_idx + num_cols_to_process
                                    if f >= self.Sc - i * self.num_out_ch_per_group - out_ch_step * self.num_filters_per_step:
                                        break

                                    if self.ifm_mm == 'unicast':
                                        if c < num_rows_to_process:
                                            this_ifm = ifm_to_process[c]
                                            this_ofm = ofm_to_process[c][f]
                                            # count how many MACs are used
                                            mac_used += num_cols_to_process
                                            # Unlike other convolutions, inside each core, IFM mode is broadcast
                                            this_ifm_demand[c][start_idx: end_idx] = this_ifm
                                            this_ofm_demand[c][start_idx // self.arr_col] = this_ofm
                                    elif self.ifm_mm == 'broadcast':
                                        # There are less than 'num_core' filters
                                        if c >= len(ofm_to_process):
                                            break
                                        this_ifm = ifm_to_process[0]
                                        mac_used += num_cols_to_process
                                        this_ifm_demand[c][start_idx: end_idx] = this_ifm
                                        this_ofm_demand[c][start_idx // self.arr_col] = ofm_to_process[c][f]
                                    else:
                                        raise ValueError("Currently only support unicast/broadcast IFM mapping mode")
                            # flatten to 1D
                            this_ifm_demand = [j for sub in this_ifm_demand for j in sub]
                            this_ofm_demand = [j for sub in this_ofm_demand for j in sub]

                            # calculate the overall utilization of the current compute cycle
                            this_mac_util = mac_used / self.num_pe

                            ifmap_compute_matrix.append(this_ifm_demand)
                            ofmap_demand_matrix.append(this_ofm_demand)
                            this_set = {s for s in this_ifm_demand}
                            ifmap_compute_set.append(this_set)
                            self.compute_utility_per_fold.append(this_mac_util)

        # ==============================================================
        # Create IFM SRAM demand matrix
        # ==============================================================

        # remove reused elements for each cycle because our NPU can ensure that reused elements stay in buffer
        # ifmap_demand_matrix = copy.deepcopy(ifmap_compute_matrix)  # too slow
        ifmap_demand_matrix = [[-1 for _ in range(self.num_pe)] for _ in range(len(ifmap_compute_matrix))]
        for row_idx in range(len(ifmap_compute_matrix)):
            if row_idx == 0:
                continue
            np_row = np.array(ifmap_compute_matrix[row_idx])
            _, unique_idx = np.unique(np_row, return_index=True)
            for idx in unique_idx:
                if ifmap_compute_matrix[row_idx][idx] == -1:
                    continue
                if ifmap_compute_matrix[row_idx][idx] not in ifmap_compute_set[row_idx - 1]:
                    ifmap_demand_matrix[row_idx][idx] = ifmap_compute_matrix[row_idx][idx]

        self.ifmap_compute_matrix = np.array(ifmap_compute_matrix)
        self.ifmap_demand_matrix = np.array(ifmap_demand_matrix)
        self.ofmap_demand_matrix = np.array(ofmap_demand_matrix)

    # Filter demand matrix
    def create_filter_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        filter_demand_matrix = []
        inter_fold_gap_postfix_mat = (np.ones((1, self.num_pe)) * -1).tolist()[0]

        for i in range(self.filter_fold):
            for out_r in range(self.out_row_fold):
                for out_c in range(self.out_col_fold):
                    for j in range(self.out_ch_fold):
                        for k in range(self.window_fold):
                            start_col_idx = i * self.num_out_ch_per_group + j * self.num_filters_per_step
                            start_in_ch = k * self.num_in_ch_per_step

                            # ==============================================================
                            # Prepare Filter data
                            # ==============================================================

                            filter_to_process = []
                            used_cores = 0
                            if self.filter_mm == 'unicast':
                                used_cores = self.num_core
                            elif self.filter_mm == 'broadcast':
                                used_cores = 1
                            for c in range(used_cores):
                                this_filter_col = []
                                for f in range(self.num_filters_per_core):
                                    col_idx = start_col_idx + c * self.num_filters_per_core + f
                                    if col_idx >= self.Sc:
                                        break
                                    filter_col = []
                                    for p in range(self.filter_size):
                                        row_idx = start_in_ch + p * self.num_in_ch
                                        max_row_idx = p * self.num_in_ch + self.num_in_ch
                                        end_row_idx = min(start_in_ch + p * self.num_in_ch + self.num_in_ch_per_step, max_row_idx)
                                        this_ = self.filter_op_mat[row_idx: end_row_idx, col_idx].reshape((1, -1)).tolist()[0]
                                        filter_col += this_
                                    this_filter_col.append(filter_col)
                                if len(this_filter_col) != 0:
                                    filter_to_process.append(this_filter_col)

                            this_fold_demand = (np.ones((self.num_core, self.arr_row * self.arr_col)) * -1).tolist()

                            num_rows_to_process = len(filter_to_process)
                            if not any(filter_to_process):
                                break
                            num_cols_to_process_per_core = len(filter_to_process[0][0])

                            # for row in filter_to_process:
                            #     assert len(row) == num_cols_to_process_per_core, 'Filter demand matrix is not a rectangle'

                            # ==============================================================
                            # Fill MAC cores
                            # ==============================================================

                            for c in range(self.num_core):
                                for f in range(self.num_filters_per_core):
                                    start_idx = f * self.num_rows_per_filter * self.arr_col
                                    end_idx = start_idx + num_cols_to_process_per_core

                                    if self.filter_mm == 'unicast':
                                        if c >= num_rows_to_process:
                                            break
                                        this_filter = filter_to_process[c][f]
                                        this_fold_demand[c][start_idx: end_idx] = this_filter
                                    elif self.filter_mm == 'broadcast':
                                        this_filter = filter_to_process[0][f]
                                        this_fold_demand[c][start_idx: end_idx] = this_filter
                                    else:
                                        raise ValueError("Currently only support unicast/broadcast filter mapping mode")
                            this_fold_demand = [j for sub in this_fold_demand for j in sub]
                            filter_demand_matrix.append(this_fold_demand)

                            for step in range(self.width_step - 1):
                                if out_c * (self.num_out_w_per_step * self.width_step) + (step + 1) * self.num_out_w_per_step >= self.out_col:
                                    break
                                filter_demand_matrix.append(inter_fold_gap_postfix_mat)
        filter_demand_matrix.append(inter_fold_gap_postfix_mat)
        self.filter_demand_matrix = np.array(filter_demand_matrix)

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

    def get_ifmap_compute_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_compute_matrix

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

    def get_compute_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()
        
        return self.ifmap_compute_matrix

    def get_num_filters_per_core(self):
        assert self.params_set_flag, 'Parameters not set yet'

        return self.num_filters_per_core

    def print_ofmap_trace(self, filename):
        assert self.demand_mat_ready_flag, 'Traces not generated yet'
        np.savetxt(filename, self.ofmap_demand_matrix, fmt='%d', delimiter=",")

    def print_ifmap_trace(self, filename):
        assert self.demand_mat_ready_flag, 'Traces not generated yet'
        np.savetxt(filename, self.ifmap_demand_matrix, fmt='%d', delimiter=",")

    def print_filter_trace(self, filename):
        assert self.demand_mat_ready_flag, 'Traces not generated yet'
        np.savetxt(filename, self.filter_demand_matrix, fmt='%d', delimiter=",")

