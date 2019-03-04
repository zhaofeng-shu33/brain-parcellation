# relies on installation of info_cluster.py
# see https://github.com/zhaofeng-shu33/principal_sequence_of_partition
import numpy as np
import nibabel as nib
from nilearn import plotting

from info_cluster import InfoCluster
from weight_builder import Builder
from loader import MultiBrain
class BrainCluster(InfoCluster):
    def __init__(self):
        self.builder = Builder()
        self.weight_matrix = (self.builder.weight_matrix + self.builder.weight_matrix.T)/2
        self.multi_brain = MultiBrain()
        self.multi_brain.build_node()
        super(BrainCluster, self).__init__(affinity = 'precomputed')    
    def visualize(self, num_partition):
        try:
            self.partition_num_list
        except AttributeError as e:
            self.fit(self.weight_matrix)
        cat = self.get_category(num_partition) # list
        self.vis_array = np.array(self.multi_brain.mask, dtype='int', copy=True)
        size = self.multi_brain.GRID_SIZE
        for i in range(len(cat)):
            x,y,z = self.multi_brain.nodes[i]
            self.vis_array[x:x+size,y:y+size,z:z+size] = self.multi_brain.mask[x:x+size,y:y+size,z:z+size] *(cat[i]+1)
        self.img = nib.Nifti1Image(self.vis_array, np.eye(4))
        plotting.plot_roi(self.img)
        plotting.show()