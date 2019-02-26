import os
from scipy.io import loadmat
class MultiBrain:
    DIR_ROOT = 'H:/'
    X_DIM = 61
    Y_DIM = 73
    Z_DIM = 61
    T_DIM = 212
    FILE_EXTENSION = 'mat'
    def __init__(self, **kargs):
        for k,v in kargs.items():
            setattr(self, k, v)
        self.file_list = []
        for item in os.listdir(self.DIR_ROOT):
            first_structure = os.path.join(self.DIR_ROOT, item)
            if not(os.path.isdir(first_structure)):
                continue
            for mat_file in os.listdir(first_structure):
                if(mat_file.endswith(self.FILE_EXTENSION)):
                    self.file_list.append(os.path.join(first_structure, mat_file))
        # load mask file
        
    def get_time_series(self, i, x, y, z):
        file_name = self.file_list[i]
        mat_dic = loadmat(file_name, variable_names='simg')
        xyzt_array = mat_dic['simg']
        return xyzt_array[x, y, z, :]
    