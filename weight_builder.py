from ace_nn import ace_nn
import loader
import reducer
import numpy as np
import os
import time
import threading
import psutil
class Builder:
    NUM_NODES = 141
    CACHE_DIR = 'F:/cache/'
    ENABLE_CACHE = True
    def worker(self, i, j):
        start_time = time.time()
        self.weight_matrix[i, j] = ace_nn(self.data[:,:,i], self.data[:,:,j], return_hscore=True)
        current_time_used = '%.2f'%(time.time() - start_time)
        print('{0}/{1}, time used: {2}s'.format(self.cnt, self.total_computation_time, current_time_used))      
    def __init__(self):
        # get the size (num_of_features, num_of_instances)
        weight_file = os.path.join(self.CACHE_DIR, 'weight_matrix.npy')
        if(os.path.exists(weight_file)):
            self.weight_matrix = np.load(weight_file)
            return
        data_0 = self.get_data(0)
        nof, noi = data_0.shape
        self.data = np.zeros(shape=(nof,noi,self.NUM_NODES))
        self.data[:,:,0] = data_0
        del data_0
        for i in range(1, self.NUM_NODES):
            self.data[:,:,i] = self.get_data(i)
        # use a numpy array to store the weight
        print('loading data finished')
        self.weight_matrix = np.zeros(shape=(self.NUM_NODES, self.NUM_NODES))
        self.total_computation_time = (self.NUM_NODES * (self.NUM_NODES - 1))/2
        self.cnt = 1
        proc=psutil.Process(os.getpid())
        for i in range(self.NUM_NODES):
            for j in range(i+1, self.NUM_NODES):
                t = threading.Thread(target=self.worker,args=(i, j))
                t.start()
                t.join() 
                print('rss: %d' % proc.memory_info().rss)
                self.cnt += 1
        if(self.ENABLE_CACHE):
            np.save(weight_file, self.weight_matrix)
    def get_data(self, index):
        data_file = os.path.join(self.CACHE_DIR, 'feature-node-reduce-{0}.npy'.format(index))
        if not(os.path.exists(data_file)):
            pre_data_file = os.path.join(self.CACHE_DIR, 'feature-node-{0}.npy'.format(index))
            if not(os.path.exists(pre_data_file)):
                a = loader.MultiBrain()
            else:
                a = None
            r = reducer.Reducer(a, index)
            r.reduce()
            return r.data_reduced
        else:
            return np.load(data_file)