# relies on installation of info_cluster.py
# see https://github.com/zhaofeng-shu33/principal_sequence_of_partition
from info_cluster import InfoCluster
from weight_builder import Builder
class BrainCluster(InfoCluster):
    def __init__(self):
        self.builder = Builder()
        self.weight_matrix = (self.builder.weight_matrix + self.builder.weight_matrix.T)/2
        super(BrainCluster, self).__init__(affinity = 'precomputed')    
