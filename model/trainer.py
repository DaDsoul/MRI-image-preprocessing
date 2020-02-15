import numpy as np
import itertools
from random import shuffle

class Trainer: 


    def __init__(self, config):
        self.config = config




    def geneterator(self, data, indices, batch_size = 1, n_labels = 1, patch_shape = None, patch_start_offset = None, augment = False):
        
        old_indices = indices 

        while True: 
            x_list = list()
            y_list = list()


            # the last three numbers are the height, width and the depth of the MRI scan
            image_shape = data.root.data.shape[-3:]

            if patch_shape: 
                indices = self.create_patch_index_list(indices, image_shape, patch_shape)
            else:
                indices = np.array(old_indices)

            shuffle(indices)


            while len(indices) > 0:
                index = indices.pop()
                




        
        pass


    @staticmethod 
    def compute_patch_indices(image_shape, patch_shape, start):
        if start is None: 
            start = [0,0,0]
        stop = image_shape + start 
        step = patch_shape
        return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)
        

    def create_patch_index_list(self, indices, image_shape, patch_shape, patch_start_offet = None):

        patch_index = list()

        for index in indices:
            if patch_start_offet is not None:
                start_offset = np.negative(tuple([np.random.choice(patch_start_offet[index] + 1) for index in range(len(patch_start_offet))]))
                patches = self.compute_patch_indices(image_shape, patch_shape, start_offset)
            else:
                patches = self.compute_patch_indices(image_shape, patch_shape, patch_start_offet)
            
            # extend add each input element, treating it seperately
            patch_index.extend(itertools.product([index], patches))


        return patch_index

    