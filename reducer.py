import time
import numpy as np
import os
class Reducer:
    '''
        use pca technique to reduce the dimension of features
        for concatenated brain data
    '''
    TARGET_DIMENSION = 500
    CACHE_DIR = 'F:/cache/'
    ENABLE_CACHE = True
    def __init__(self, loader, node_index, force_rewrite = False):
        '''
            loader : MultiBrain
            node_index : int
            force_rewrite : Boolean
        '''
        self.loader = loader
        self.node_index = node_index
        # try to load from cache
        self.data_file = os.path.join(self.CACHE_DIR, 'feature-node-{0}.npy'.format(self.node_index))
        if(os.path.exists(self.data_file) and force_rewrite == False):
            self.data = np.load(self.data_file)
        else:
            # construct an array with (n_samples, n_features)
            x, y, z = self.loader.nodes[self.node_index]
            start_time = time.time()
            self.data = self.loader.get_time_series(0, x, y, z)
            for i in range(1, self.loader.num_instances):
                new_data = self.loader.get_time_series(i, x, y, z)
                self.data = np.vstack((self.data, new_data))
                current_time_used = int(time.time() - start_time)
                print('{0}/{1}, time used: {2}s'.format(i, self.loader.num_instances, current_time_used))
            if(self.ENABLE_CACHE):
                np.save(self.data, self.data_file)
        