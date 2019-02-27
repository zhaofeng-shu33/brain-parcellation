import time
import numpy as np
class Reducer:
    '''
        use pca technique to reduce the dimension of features
        for concatenated brain data
    '''
    TARGET_DIMENSION = 500
    def __init__(self, loader, node_index):
        '''
            loader : MultiBrain
            node_index : int
        '''
        self.loader = loader
        self.node_index = node_index
        # construct an array with (n_samples, n_features)
        x, y, z = self.loader.nodes[self.node_index]
        start_time = time.time()
        self.data = self.loader.get_time_series(0, x, y, z)
        for i in range(1, self.loader.num_instances):
            new_data = self.loader.get_time_series(i, x, y, z)
            self.data = np.vstack((self.data, new_data))
            current_time_used = int(time.time() - start_time)
            print('{0}/{1}, time used: {2}s'.format(i, self.loader.num_instances, current_time_used))
            