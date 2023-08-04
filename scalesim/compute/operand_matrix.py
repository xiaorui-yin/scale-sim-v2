import math
import numpy as np
from tqdm import tqdm

from scalesim.topology_utils import topologies as topoutil
from scalesim.scale_config import scale_config as cfg


# This class defines data types for operand matrices
class operand_matrix(object):
    def __init__(self):
        # Objects from outer container classes
        self.config = cfg()
        self.topoutil = topoutil()

        # Layer hyper parameters
        self.layer_id = 0
        self.ifmap_rows, self.ifmap_cols = 1, 1
        self.filter_rows, self.filter_cols = 1, 1
        self.num_input_channels, self.num_filters = 1, 1
        self.row_stride, self.col_stride = 1, 1
        self.pad_t, self.pad_r, self.pad_b, self.pad_l = 0, 0, 0, 0
        self.dw_flag = False
        self.batch_size = 1

        #  Derived hyper parameters
        self.ofmap_px_per_filt, self.conv_window_size = 1, 1
        self.ofmap_rows, self.ofmap_cols = 1, 1

        # Offsets
        self.ifmap_offset, self.filter_offset, self.ofmap_offset = 0, 10000000, 20000000
        self.matrix_offset_arr = [0, 10000000, 20000000]

        # Address matrices
        self.ifmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.conv_window_size), dtype=int)
        self.filter_addr_matrix = np.ones((self.conv_window_size, self.num_filters), dtype=int)
        self.ofmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.num_filters), dtype=int)

        # Flags
        self.params_set_flag = False
        self.matrices_ready_flag = False

    #
    def set_params(self,
                   config_obj,
                   topoutil_obj,
                   layer_id=0,
                   ):

        self.config = config_obj
        self.topoutil = topoutil_obj
        self.layer_id = layer_id

        # TODO: Marked for cleanup
        #my_name = 'operand_matrix.set_params(): '
        #err_prefix = 'Error: ' + my_name
        #
        #if (not len(layer_hyper_param_arr) == 7 and not len(layer_hyper_param_arr) == 8
        #        and not len(layer_hyper_param_arr) == 9) or (not len(layer_calc_hyper_param_arr) == 4) \
        #        or (not len(self.matrix_offset_arr) == 3):
        #    message = err_prefix + 'Invalid arguments. Exiting.'
        #    print(message)
        #    return -1

        self.ifmap_rows, self.ifmap_cols = self.topoutil.get_layer_ifmap_dims(self.layer_id)
        self.filter_rows, self.filter_cols = self.topoutil.get_layer_filter_dims(self.layer_id)
        self.num_input_channels = self.topoutil.get_layer_num_channels(self.layer_id)
        self.num_filters = self.topoutil.get_layer_num_filters(self.layer_id)
        self.row_stride, self.col_stride = self.topoutil.get_layer_strides(self.layer_id)
        paddings = self.topoutil.get_layer_paddings(self.layer_id)
        self.pad_t, self.pad_r, self.pad_b, self.pad_l = paddings[0], paddings[1], paddings[2], paddings[3]
        self.dw_flag = self.topoutil.get_layer_dw_flag(self.layer_id)

        # TODO: Marked for cleanup
        #self.row_stride = layer_hyper_param_arr[6]
        #if len(layer_hyper_param_arr) == 8:
        #    self.col_stride = layer_hyper_param_arr[7]

        # TODO: Anand
        # TODO: Next release
        # TODO: Add an option for batching
        self.batch_size = 1

        # TODO: Marked for cleanup
        #if len(layer_hyper_param_arr) == 9:
        #    self.batch_size = layer_hyper_param_arr[8]

        # Assign the calculated hyper parameters
        self.ofmap_rows, self.ofmap_cols = self.topoutil.get_layer_ofmap_dims(self.layer_id)
        self.ofmap_rows = int(self.ofmap_rows)
        self.ofmap_cols = int(self.ofmap_cols)

        self.ofmap_px_per_filt = int(self.ofmap_rows * self.ofmap_cols)
        if self.dw_flag:
            self.ofmap_px_per_filt = int(self.ofmap_rows * self.ofmap_cols * self.num_input_channels)

        self.conv_window_size = int(self.topoutil.get_layer_window_size(self.layer_id))

        # Assign the offsets
        self.ifmap_offset, self.filter_offset, self.ofmap_offset \
            = self.config.get_offsets()

        # Address matrices: This is needed to take into account the updated dimensions
        self.ifmap_addr_matrix = np.ones((self.ofmap_px_per_filt * self.batch_size, self.conv_window_size), dtype='>i4')
        self.filter_addr_matrix = np.ones((self.conv_window_size, self.num_filters), dtype='>i4')
        self.ofmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.num_filters), dtype='>i4')

        self.params_set_flag = True

        # TODO: This should be called from top level
        # TODO: Implement get() function for getting the matrix
        # TODO: Marked for cleanup
        # Return 0 if operand matrix generation is successful
        #self.create_operand_matrices()
        #if self.matrices_ready_flag:
        #    return True, self.ifmap_addr_matrix, self.filter_addr_matrix, self.ofmap_addr_matrix
        #else:
        #    message = err_prefix + 'Address Matrices not created. Exiting!'
        #    print(message)
        #    return False, None, None, None

    # top level function to create the operand matrices
    def create_operand_matrices(self):
        my_name = 'operand_matrix.create_operand_matrices(): '
        err_prefix = 'Error: ' + my_name

        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1

        retcode_1 = self.create_ifmap_matrix()
        retcode_2 = self.create_filter_matrix()
        retcode_3 = self.create_ofmap_matrix()

        retcode = retcode_1 + retcode_2 + retcode_3
        if retcode == 0:
            self.matrices_ready_flag = True

        return retcode

    # creates the ifmap operand
    def create_ifmap_matrix(self):
        my_name = 'operand_matrix.create_ifmap_matrix(): '
        err_prefix = 'Error: ' + my_name

        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1

        # for row_idx in tqdm(range(self.batch_size*self.ofmap_px_per_filt)):
        for row_idx in range(self.batch_size*self.ofmap_px_per_filt):
            for col_idx in range(self.conv_window_size):
                self.ifmap_addr_matrix[row_idx][col_idx] = self.calc_ifmap_elem_addr(i=row_idx, j=col_idx)
        return 0

    # For each convolution window, calculate the index of the IFM element in HWC format
    # i is the window index
    # j is the element index within the window
    def calc_ifmap_elem_addr(self, i, j):
        ifmap_cols = self.ifmap_cols
        filter_col = self.filter_cols
        filter_row = self.filter_rows
        r_stride = self.row_stride
        c_stride = self.col_stride
        Ew = self.ofmap_cols
        channel = self.num_input_channels

        ofm_dims = (self.ofmap_rows, self.ofmap_cols)
        filter_dims = (filter_row, filter_col)
        stride = (r_stride, c_stride)
        padding = (self.pad_t, self.pad_r, self.pad_b, self.pad_l)

        if not self.dw_flag:
            # Calculate the row and col in the Eh X Ew mat
            ofmap_row = int(math.floor(i / Ew))
            ofmap_col = int(i % Ew)

            # which channel
            channel_idx = int(j % channel)
            # which row within the window
            row_idx = int(j // (filter_col * channel))
            # which col within the window
            col_idx = int((j % (filter_col * channel)) // channel)

            # If this is the padded position
            ofm_idx = (ofmap_row, ofmap_col)
            window_idx = (row_idx, col_idx)
            if is_padded(ofm_dims, ofm_idx, window_idx, padding, filter_dims, stride):
                return -1
            else:
                # Change this to corresponding ifmap row col for the start of the conv window
                i_row = ofmap_row * r_stride - self.pad_t + row_idx
                i_col = ofmap_col * c_stride - self.pad_l + col_idx
                return i_row * ifmap_cols * channel + i_col * channel + channel_idx
        else:
            # depthwise conv

            # Calculate the row and col in the Eh X Ew mat
            ofmap_row = int(math.floor(i / (Ew * channel)))
            ofmap_col = int((i % (Ew * channel)) // channel)

            # which channel
            channel_idx = int(i % channel)
            # which row within the window
            row_idx = int(j // filter_col)
            # which col within the window
            col_idx = int(j % filter_col)

            # If this is the padded position
            ofm_idx = (ofmap_row, ofmap_col)
            window_idx = (row_idx, col_idx)
            if is_padded(ofm_dims, ofm_idx, window_idx, padding, filter_dims, stride):
                return -1
            else:
                # Change this to corresponding ifmap row col for the start of the conv window
                i_row = ofmap_row * r_stride - self.pad_t + row_idx
                i_col = ofmap_col * c_stride - self.pad_l + col_idx
                return i_row * ifmap_cols * channel + i_col * channel + channel_idx

    # creates the ofmap operand
    def create_ofmap_matrix(self):
        my_name = 'operand_matrix.create_ofmap_matrix(): '
        err_prefix = 'Error: ' + my_name
        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1
        # for row_idx in tqdm(range(self.ofmap_px_per_filt)):
        for row_idx in range(self.ofmap_px_per_filt):
            for col_idx in range(self.num_filters):
                self.ofmap_addr_matrix[row_idx][col_idx] = self.calc_ofmap_elem_addr(i=row_idx, j=col_idx)
        return 0

    # logic to translate ofmap into matrix resulting systolic array MACs
    def calc_ofmap_elem_addr(self, i, j):
        num_filt = self.num_filters
        internal_address = num_filt * i + j
        ofmap_px_addr = internal_address
        return ofmap_px_addr

    # creates the filter operand
    def create_filter_matrix(self):
        my_name = 'operand_matrix.create_filter_matrix(): '
        err_prefix = 'Error: ' + my_name
        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1
        # for row_idx in tqdm(range(self.conv_window_size)):
        for row_idx in range(self.conv_window_size):
            for col_idx in range(self.num_filters):
                self.filter_addr_matrix[row_idx][col_idx] = self.calc_filter_elem_addr(i=row_idx, j=col_idx)

        return 0

    # logic to translate filter into matrix fed into systolic array MACs
    def calc_filter_elem_addr(self, i, j):
        filter_row = self.filter_rows
        filter_col = self.filter_cols
        channel = self.num_input_channels

        if not self.dw_flag:
            internal_address = j * filter_row * filter_col * channel + i
            filter_px_addr = internal_address
        else:
            internal_address = j * filter_row * filter_col + i
            filter_px_addr = internal_address
        return filter_px_addr

    # function to get a part or the full ifmap operand
    def get_ifmap_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
                              num_cols=-1):
        if num_rows == -1:
            num_rows = self.ofmap_px_per_filt
        if num_cols == -1:
            num_cols = self.conv_window_size
        my_name = 'operand_matrix.get_ifmap_matrix_part(): '
        err_prefix = 'Error: ' + my_name
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
                print(message)
                return -1, np.zeros((1, 1))
        if (start_row + num_rows) > self.ofmap_px_per_filt or (start_col + num_cols) > self.conv_window_size:
            message = err_prefix + ": Illegal arguments. Exiting!"
            print(message)
            return -2, np.zeros((1, 1))

        # Anand: ISSUE #3. Patch
        #end_row = start_row + num_rows + 1
        #end_col = start_col + num_cols + 1
        #ret_mat = self.ifmap_addr_matrix[start_row: end_row][start_col: end_col]
        end_row = start_row + num_rows
        end_col = start_col + num_cols
        ret_mat = self.ifmap_addr_matrix[start_row: end_row, start_col: end_col]
        return 0, ret_mat

    def get_ifmap_matrix(self):
        return self.get_ifmap_matrix_part()

    # function to get a part or the full filter operand
    def get_filter_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
                               num_cols=-1):

        if num_rows == -1:
            num_rows = self.conv_window_size
        if num_cols == -1:
            num_cols = self.num_filters
        my_name = 'operand_matrix.get_filter_matrix_part(): '
        err_prefix = 'Error: ' + my_name
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
                print(message)
                return -1, np.zeros((1, 1))
        if (start_row + num_rows) > self.conv_window_size or (start_col + num_cols) > self.num_filters:
            message = err_prefix + ": Illegal arguments. Exiting!"
            print(message)
            return -2, np.zeros((1, 1))

        # Anand: ISSUE #3. FIX
        #end_row = start_row + num_rows + 1
        #end_col = start_col + num_cols + 1
        end_row = start_row + num_rows
        end_col = start_col + num_cols

        # Anand: ISSUE #3. FIX
        #ret_mat = self.filter_addr_matrix[start_row: end_row][start_col: end_col]
        ret_mat = self.filter_addr_matrix[start_row: end_row, start_col: end_col]
        return 0, ret_mat

    def get_filter_matrix(self):
        return self.get_filter_matrix_part()

    # function to get a part or the full ofmap operand
    def get_ofmap_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
                               num_cols=-1):

        # Since we cannot pass self as an argument in the member functions
        # This is an alternate way of making the matrix dimensions as defaults
        if num_rows == -1:
            num_rows = self.ofmap_px_per_filt
        if num_cols == -1:
            num_cols = self.num_filters
        my_name = 'operand_matrix.get_ofmap_matrix_part(): '
        err_prefix = 'Error: ' + my_name
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
                print(message)
                return -1, np.zeros((1, 1))
        if (start_row + num_rows) > self.ofmap_px_per_filt or (start_col + num_cols) > self.num_filters:
            message = err_prefix + ": Illegal arguments. Exiting!"
            print(message)
            return -2, np.zeros((1, 1))

        # Anand: ISSUE #3. Patch
        #end_row = start_row + num_rows + 1
        #end_col = start_col + num_cols + 1
        #ret_mat = self.filter_addr_matrix[start_row: end_row][start_col: end_col]
        end_row = start_row + num_rows
        end_col = start_col + num_cols
        # Anand: ISSUE #7. Patch
        #ret_mat = self.filter_addr_matrix[start_row: end_row, start_col: end_col]
        ret_mat = self.ofmap_addr_matrix[start_row: end_row, start_col: end_col]

        return 0, ret_mat

    def get_ofmap_matrix(self):
        return self.get_ofmap_matrix_part()

    def get_all_operand_matrix(self):
        if not self.matrices_ready_flag:
            me = 'operand_matrix.' + 'get_all_operand_matrix()'
            message = 'ERROR:' + me + ': Matrices not ready or matrix gen failed'
            print(message)
            return

        return self.ifmap_addr_matrix, \
               self.filter_addr_matrix, \
               self.ofmap_addr_matrix


def is_padded(ofm_dim, ofm_idx, window_idx, pad, filter_size, stride):
    ofm_rows, ofm_cols = ofm_dim
    ofm_row_idx, ofm_col_idx = ofm_idx
    window_row_idx, window_col_idx = window_idx
    pad_t, pad_r, pad_b, pad_l = pad
    row_stride, col_stride = stride
    filter_row, filter_col = filter_size

    # Top padding
    if ofm_row_idx < math.ceil(pad_t / row_stride):
        if window_row_idx < pad_t - (ofm_row_idx * row_stride):
            return True

    # Bottom padding
    if ofm_row_idx >= ofm_rows - math.ceil(pad_b / row_stride):
        if window_row_idx >= filter_row - (pad_b - (ofm_rows - ofm_row_idx - 1) * row_stride):
            return True

    # Left padding
    if ofm_col_idx < math.ceil(pad_l / col_stride):
        if window_col_idx < pad_l - (ofm_col_idx * col_stride):
            return True

    # Right padding
    if ofm_col_idx >= ofm_cols - math.ceil(pad_r / col_stride):
        if window_col_idx >= filter_col - (pad_r - (ofm_cols - ofm_col_idx - 1) * col_stride):
            return True

    return False


if __name__ == '__main__':
    opmat = operand_matrix()
    tutil = topoutil()
    lid = 3
    topology_file = "../../topologies/mlperf/test.csv"
    tutil.load_arrays(topofile=topology_file)
    for i in range(tutil.get_num_layers()):
        layer_param_arr = tutil.get_layer_params(layer_id=i)
        ofmap_dims = tutil.get_layer_ofmap_dims(layer_id=i)
        ofmap_px_filt = tutil.get_layer_num_ofmap_px(layer_id=i) / tutil.get_layer_num_filters(layer_id=i)
        conv_window_size = tutil.get_layer_window_size(layer_id=i)
        layer_calc_hyper_param_arr = [ofmap_dims[0], ofmap_dims[1], ofmap_px_filt, conv_window_size]
        config_arr = [512, 512, 256, 8, 8]
        #[matrix_set, ifmap_addr_matrix, filter_addr_matrix, ofmap_addr_matrix] \
        #    = opmat.set_params(layer_hyper_param_arr=layer_param_arr[1:],
        #                       layer_calc_hyper_param_arr=layer_calc_hyper_param_arr,
        #                       offset_list=[0, 1000000, 2000000])
