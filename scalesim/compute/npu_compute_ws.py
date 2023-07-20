import math
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg


class npu_compute_ws:
    def __init__(self):
        # Params set by user
        self.config = cfg()

        self.ifmap_op_mat = np.zeros((1, 1))
        self.ofmap_op_mat = np.zeros((1, 1))
        self.filter_op_mat = np.zeros((1, 1))

        self.ifm_mm = 'broadcast'
        self.filter_mm = 'broadcast'
        self.dw_flag = False

        # Derived parameters
        self.Sr = 0
        self.Sc = 0
        self.T = 0

        self.out_col = 0

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

        self.num_out_px_per_step = 0
        self.out_px_fold = 0

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

    #
    def set_params(self,
                   config_obj=cfg(),
                   ifm_mm='broadcast',
                   filter_mm='broadcast',
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

        self.out_col = out_col

        ifmap_col = self.ifmap_op_mat.shape[1]
        filter_row = self.filter_op_mat.shape[0]

        assert ifmap_col == filter_row, "Dimension mismatch between operands"

        self.Sr = self.ifmap_op_mat.shape[1]  # window size k_h * k_w * c_i
        self.Sc = self.filter_op_mat.shape[1]  # c_o
        self.T = self.ifmap_op_mat.shape[0]  # number of output pixels

        self.num_core, self.arr_row, self.arr_col = self.config.get_array_dims()
        self.num_pe = self.num_core * self.arr_row * self.arr_col

        self.num_rows_per_filter = int(math.pow(2, math.ceil(math.log2(self.Sr / self.arr_col))))
        assert self.num_rows_per_filter <= self.arr_row, 'Single filter cannot exceed one core'  # TODO first_layer_7x7

        self.num_filters_per_core = int(self.arr_row // self.num_rows_per_filter)
        assert (self.arr_row / self.num_rows_per_filter).is_integer(), "Must be integer"

        if self.filter_mm == 'unicast':
            self.num_filters_per_step = self.num_core * self.num_filters_per_core
        elif self.filter_mm == 'double':
            self.num_filters_per_step = 2 * self.num_filters_per_core
        elif self.filter_mm == 'broadcast':
            self.num_filters_per_step = self.num_filters_per_core
        else:
            raise ValueError("Mapping Mode must be {'unicast', 'broadcast', 'double'}")
        self.filter_fold = math.ceil(self.Sc / self.num_filters_per_step)

        if self.ifm_mm == 'broadcast':
            self.num_out_px_per_step = 1
        elif self.ifm_mm == 'unicast':
            self.num_out_px_per_step = self.num_core * self.num_filters_per_core
        elif self.ifm_mm == 'double':  # TODO
            self.num_out_px_per_step = self.num_core * self.num_filters_per_core / 2
        else:
            raise ValueError("Mapping Mode must be {'unicast', 'broadcast', 'double'}")
        if self.dw_flag:
            self.out_px_fold = math.ceil((self.T / self.num_out_px_per_step) / self.filter_fold)
        else:
            self.out_px_fold = math.ceil(self.T / self.num_out_px_per_step)

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

        self.demand_mat_ready_flag = True
    
    # IFM demand matrix
    def create_ifmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        # clock cycles for weights to be loaded (one clock cycle)
        inter_fold_gap_prefix_mat = (np.ones((1, self.num_pe)) * -1).tolist()[0]
        ifmap_demand_matrix = [inter_fold_gap_prefix_mat]
        
        for i in range(self.filter_fold):
            for j in range(int(self.out_px_fold)):
                # ==============================================================
                # Prepare IFM data
                # ==============================================================
                if self.dw_flag:
                    if self.Sr == 9:
                        # depthwise 3x3: work on 2x2 output pixels at a time
                        num_horizontal_pixels = int((self.num_out_px_per_step / self.num_filters_per_step) / 2)
                        # num_vertical_pixels == 2 (fixed)
                        # ifm_addr_mat:
                        # [px0_ch0]
                        # [px0_ch1]
                        # [px0_...]
                        # [px0_chC]
                        # [px1_ch0]
                        # [px1_ch1]
                        # [px1_...]
                        # [px1_chC]
                        # [px2_...]
                        # num rows = out_row * out_col * chC
                        # assume num_filters_per_step = 16, out_col = 56, num_ch = Sc = 32
                        # first time we work on px0(ch0 ~ ch15), px1(ch0 ~ ch15), px56(ch0 ~ ch15), px57(ch0 ~ ch15) start_row_idx = 0
                        # move to next 2x2 output pixels, we work on px2(ch0 ~ ch15), px3(ch0 ~ ch15), px58(ch0 ~ ch15), px59(ch0 ~ ch15) start_row_idx = 2 * 32 = 64
                        # once first two rows are done, we move to the third row, start_row_idx = 2 * 56 * 32 = 3584
                        start_row_idx = int(j % (self.out_col / 2)) * num_horizontal_pixels * self.Sc +\
                                        int(j / (self.out_col / 2)) * self.out_col * self.Sc * 2
                    elif self.Sr == 25:
                        # depthwise 5x5: work on 1x1 output pixels at a time
                        start_row_idx = j * self.Sc
                else:
                    start_row_idx = j * self.num_out_px_per_step
                end_row_idx = min(start_row_idx + self.num_out_px_per_step, self.T)

                # Remaining filters to be processed

                filter_delta = self.Sc - i * self.num_filters_per_step
                # Remaining output pixels to be processed
                out_px_delta = self.T - start_row_idx

                # IFM data to load
                ifm_to_process = []
                if self.dw_flag:
                    num_out_px = int(self.num_out_px_per_step / self.num_filters_per_step)
                    num_ch_per_px = self.num_filters_per_step
                    for px in range(num_out_px):
                        for ch in range(num_ch_per_px):
                            # Elaboration: see how start_idx is calculated
                            row_idx = start_row_idx +\
                                math.floor(px / 2) * self.Sc * self.out_col +\
                                px % 2 * self.Sc + \
                                i * self.num_filters_per_step + ch
                            # if px == 0 and ch == 0:
                            #     ifm_to_process = self.ifmap_op_mat[row_idx, :].reshape((1, -1))
                            # else:
                            #     ifm_to_process = np.concatenate((ifm_to_process, self.ifmap_op_mat[row_idx, :].reshape((1, -1))), axis=0)
                            ifm_row = self.ifmap_op_mat[row_idx, :].tolist()
                            ifm_to_process.append(ifm_row)
                else:
                    ifm_to_process = self.ifmap_op_mat[start_row_idx: end_row_idx, :]
                    # count how many IFM elements are read (excluding padding)
                    self.ifmap_reads += np.count_nonzero(ifm_to_process != -1)
                    ifm_to_process = ifm_to_process.tolist()
                this_fold_demand = (np.ones((self.num_core, self.arr_row * self.arr_col)) * -1).tolist()

                # ==============================================================
                # Fill MAC core
                # ==============================================================
                mac_used = 0
                for c in range(self.num_core):
                    for f in range(self.num_filters_per_core):
                        start_idx = f * self.num_rows_per_filter * self.arr_col
                        end_idx = start_idx + self.Sr
                        assert end_idx <= self.arr_row * self.arr_col, "end_idx exceeds the maximum index"

                        window_idx = c * self.num_filters_per_core + f

                        if self.ifm_mm == 'broadcast':
                            if window_idx <= filter_delta:
                                mac_used += end_idx - start_idx
                                this_fold_demand[c][start_idx: end_idx] = ifm_to_process[0]
                        elif self.ifm_mm == 'unicast':
                            if window_idx <= out_px_delta:  # TODO: check this
                                mac_used += end_idx - start_idx
                                this_fold_demand[c][start_idx: end_idx] = ifm_to_process[window_idx]
                        elif self.ifm_mm == 'double':
                            # TODO
                            this_fold_demand[c][start_idx: end_idx] = ifm_to_process[(c // 2) + f]

                # calculate the overall utilization of the current compute cycle
                this_mac_util = mac_used / self.num_pe

                # this_fold_demand = this_fold_demand.reshape((1, self.num_pe))
                this_fold_demand = [j for sub in this_fold_demand for j in sub]

                ifmap_demand_matrix.append(this_fold_demand)
                self.compute_utility_per_fold.append(this_mac_util)
        self.ifmap_demand_matrix = np.array(ifmap_demand_matrix)
    #
    def create_filter_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        filter_demand_matrix = []

        if self.filter_mm == "unicast":
            num_pixels_per_step = 1
        elif self.filter_mm == "double":
            num_pixels_per_step = 2
        elif self.filter_mm == "broadcast":
            num_pixels_per_step = self.num_core
        else:
            raise ValueError("Mapping Mode must be {'unicast', 'broadcast', 'double'}")

        if self.dw_flag:
            this_T = self.T // self.Sc
        else:
            this_T = self.T

        filter_fold = self.Sc // self.num_filters_per_step
        filter_rest = self.Sc % self.num_filters_per_step

        pixel_fold = this_T // num_pixels_per_step
        pixel_rest = this_T % num_pixels_per_step

        for idx_f in range(filter_fold):
            filter_demand_row = [-1] * self.num_pe
            for paste_time in range(num_pixels_per_step):
                for opx in range(self.num_filters_per_step):
                    for wpx in range(self.Sr):
                        filter_demand_row[wpx + (opx + self.num_filters_per_step * paste_time) *
                                          self.num_rows_per_filter * self.arr_col] \
                            = wpx + (opx + idx_f * self.num_filters_per_step) * self.Sr
            filter_demand_matrix.append(filter_demand_row)
            for _ in range(pixel_fold-1):
                filter_demand_row = [-1] * self.num_pe
                filter_demand_matrix.append(filter_demand_row)
            if pixel_rest == 0:
                pass
            else:
                filter_demand_row = [-1] * self.num_pe
                filter_demand_matrix.append(filter_demand_row)

        if filter_rest == 0:
            pass
        else:
            filter_demand_row = [-1] * self.num_pe
            for paste_time in range(num_pixels_per_step):
                for opx in range(filter_rest):
                    for wpx in range(self.Sr):
                        filter_demand_row[wpx + (opx + self.num_filters_per_step * paste_time) *
                                          self.num_rows_per_filter * self.arr_col] \
                            = wpx + (opx + filter_fold * self.num_filters_per_step) * self.Sr
            filter_demand_matrix.append(filter_demand_row)
            for _ in range(pixel_fold-1):
                filter_demand_row = [-1] * self.num_pe
                filter_demand_matrix.append(filter_demand_row)
            if pixel_rest == 0:
                pass
            else:
                filter_demand_row = [-1] * self.num_pe
                filter_demand_matrix.append(filter_demand_row)

        filter_demand_row = [-1] * self.num_pe
        filter_demand_matrix.append(filter_demand_row)

        self.filter_demand_matrix = np.array(filter_demand_matrix)

    def create_ofmap_demand_mat(self):

        ofmap_demand_matrix = []

        num_row = self.num_core * self.arr_row

        if self.dw_flag:
            this_T = self.T // self.Sc
        else:
            this_T = self.T
        T_h = int(pow(this_T, 0.5))
        T_w = int(pow(this_T, 0.5))

        if self.filter_mm == "unicast":
            num_pixels_w_per_step = 1
            num_pixels_h_per_step = 1
        elif self.filter_mm == "double":
            num_pixels_w_per_step = 2
            num_pixels_h_per_step = 1
        elif self.filter_mm == "broadcast":
            num_pixels_w_per_step = 2
            num_pixels_h_per_step = self.num_core // 2
        else:
            raise ValueError("Wrong mapping mode!!!")
        num_pixels_per_step = num_pixels_w_per_step * num_pixels_h_per_step
        opx_stride = num_row // (self.num_filters_per_step * num_pixels_per_step)

        filter_fold = self.Sc // self.num_filters_per_step
        filter_rest = self.Sc % self.num_filters_per_step

        pixel_h_fold = T_h // num_pixels_h_per_step
        pixel_h_rest = T_h % num_pixels_h_per_step
        pixel_w_fold = T_w // num_pixels_w_per_step
        pixel_w_rest = T_w % num_pixels_w_per_step

        assert pixel_h_rest == 0
        assert pixel_w_rest == 0

        ofmap_demand_row = [-1] * num_row
        ofmap_demand_matrix.append(ofmap_demand_row)

        for idx_f in range(filter_fold):
            # ofmap_demand_row = [-1] * num_row
            # ofmap_demand_matrix.append(ofmap_demand_row)
            for idx_ph in range(pixel_h_fold):
                for idx_pw in range(pixel_w_fold):
                    ofmap_demand_row = [-1] * num_row
                    for opx_h in range(num_pixels_h_per_step):
                        for opx_w in range(num_pixels_w_per_step):
                            for opx_c in range(self.num_filters_per_step):
                                ofmap_demand_row[(opx_c + (opx_w + opx_h * num_pixels_w_per_step) *
                                                  self.num_filters_per_step) * opx_stride] = \
                                    opx_c + (opx_w + opx_h * T_w) * self.Sc + \
                                    idx_pw * self.Sc * num_pixels_w_per_step + \
                                    idx_ph * self.Sc * T_w * num_pixels_h_per_step + \
                                    idx_f * self.num_filters_per_step
                    ofmap_demand_matrix.append(ofmap_demand_row)

        if filter_rest == 0:
            pass
        else:
            # ofmap_demand_row = [-1] * num_row
            # ofmap_demand_matrix.append(ofmap_demand_row)
            for idx_ph in range(pixel_h_fold):
                for idx_pw in range(pixel_w_fold):
                    ofmap_demand_row = [-1] * num_row
                    for opx_h in range(num_pixels_h_per_step):
                        for opx_w in range(num_pixels_w_per_step):
                            for opx_c in range(filter_rest):
                                ofmap_demand_row[(opx_c + (opx_w + opx_h * num_pixels_w_per_step) *
                                                  self.num_filters_per_step) * opx_stride] = \
                                    opx_c + (opx_w + opx_h * T_w) * self.Sc + \
                                    idx_pw * self.Sc * num_pixels_w_per_step + \
                                    idx_ph * self.Sc * T_w * num_pixels_h_per_step + \
                                    filter_fold * self.num_filters_per_step
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