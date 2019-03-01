from ace_nn import ace_nn
from . import loader
from . import reducer
import numpy as np
class Builder:
    NUM_NODES = 141
    CACHE_DIR = 'F:/cache/'
    def __init__(self):
        # get the size (num_of_features, num_of_instances)
        data_0 = self.get_data(0)
        nof, noi = data_0.shape
        self.data = np.zeros(shape=(nof,noi,self.NUM_NODES))
        self.data[:,:,0] = data_0
        del data_0
        for i in range(1, self.NUM_NODES):
            self.data[:,:,i] = get_data(i)
        # use a numpy array to store the weight
        print('loading data finished')
        self.weight_matrix = np.zeros(shape=(self.NUM_NODES, self.NUM_NODES))
        for i in range(self.NUM_NODES):
            for j in range(i+1, self.NUM_NODES):
                self.weight_matrix[i, j] = ace_nn(self.data[:,:,i], self.data[:,:,j], return_hscore = True)
        
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