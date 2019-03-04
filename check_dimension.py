# check the four dimensions of the dataset
ROOT_DIR = 'H:/'
import os
from scipy.io.matlab.mio import mat_reader_factory
import numpy as np
def find_label(list, label):
    for item in list:
        if(item[0] == label):
            return item[1]
if __name__ == '__main__':
    os.chdir(ROOT_DIR)
    tb_cnt = 0
    tr_cnt = 0
    for tbtr in os.listdir():
        if(tbtr.find('tb') < 0 and tbtr.find('tr') < 0):
            continue
        for mat in os.listdir(tbtr):
            if(mat.find('mat')>0):
                mat_path = os.path.join(tbtr,mat)
                tmp_mio5, _ = mat_reader_factory(mat_path)
                tmp_list = tmp_mio5.list_variables()
                img_tuple = find_label(tmp_list, 'img')
                x, y, z, t = img_tuple
                assert(x == 61 and y == 73 and z == 61)
                if(t == 212):
                    assert(tbtr.find('tr') == 0)
                    tr_cnt += 1
                elif(t == 217):
                    assert(tbtr.find('tb') == 0)                    
                    tb_cnt += 1
                else:
                    print(t)
                    raise Exception("error")
    print(tr_cnt, tb_cnt)
                
