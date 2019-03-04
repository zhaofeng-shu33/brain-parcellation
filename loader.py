import os
from scipy.io import loadmat
import nibabel as nib
import numpy as np
class MultiBrain:
    '''
        used for data loader
    '''
    DIR_ROOT = 'H:/'
    X_DIM = 61
    Y_DIM = 73
    Z_DIM = 61
    T_DIM = 212
    FILE_EXTENSION = 'mat'
    MASK_FILE_NAME = 'MNI_T1_3mm_gmask.hdr'
    MASK_FILE_DIR = 'mask'
    GRID_SIZE = 10
    DIR_EXCLUDE_PATTERN = 'tb'
    def __init__(self, **kargs):
        for k,v in kargs.items():
            setattr(self, k, v)
        self.file_list = []
        self.current_mat_index = -1
        for item in os.listdir(self.DIR_ROOT):
            if(item.find(self.DIR_EXCLUDE_PATTERN)>=0):
                continue
            first_structure = os.path.join(self.DIR_ROOT, item)
            if not(os.path.isdir(first_structure)):
                continue
            for mat_file in os.listdir(first_structure):
                if(mat_file.endswith(self.FILE_EXTENSION)):
                    self.file_list.append(os.path.join(first_structure, mat_file))
        self.num_instances = len(self.file_list)
        # load mask file
        self.img = nib.load(os.path.join(self.DIR_ROOT, self.MASK_FILE_DIR, self.MASK_FILE_NAME))
        self.mask = self.img.get_fdata().astype(np.bool)
    def get_time_series(self, i, x, y, z):
        '''
            (x, y, z) is left corner
        '''
        self.change_current_mat(i)
        return self.current_mat_array[x: x+self.GRID_SIZE, y:y+self.GRID_SIZE,
            z:z+self.GRID_SIZE, :].reshape(self.GRID_SIZE**3 * self.T_DIM)
    def change_current_mat(self, i):
        if(self.current_mat_index != i):
            self.current_mat_index = i
            file_name = self.file_list[i]
            mat_dic = loadmat(file_name, variable_names='simg')            
            self.current_mat_array = mat_dic['simg']
    def build_node(self):
        '''
            graph node and array region mapping
        '''
        self.nodes = []
        for i in range(0, self.X_DIM - self.GRID_SIZE, self.GRID_SIZE):
            for j in range(0, self.Y_DIM - self.GRID_SIZE, self.GRID_SIZE):
                for k in range(0, self.Z_DIM - self.GRID_SIZE, self.GRID_SIZE):
                    if(np.sum(self.mask[i:(i+self.GRID_SIZE),
                        j:(j+self.GRID_SIZE),
                        k:(k+self.GRID_SIZE),0])>0):
                        self.nodes.append([i,j,k])
                    else:
                        print('ignore', [i,j,k])