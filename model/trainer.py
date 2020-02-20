import numpy as np
import itertools
from random import shuffle
import tables
from scipy.ndimage import zoom



class Trainer: 


    def __init__(self, config):
        self.config = config


    def get_generators(self, data_file, train_list, validation_list, patch_shape, patch_start_offset, batch_size = 6):

        train_generator = self.generator(data_file, train_list,
                                         batch_size=batch_size, 
                                         n_labels=self.config.label_num,
                                         patch_shape = patch_shape,
                                         patch_start_offset = patch_start_offset)
        
        validation_generator = self.generator(data_file, validation_list,
                                            batch_size=batch_size, 
                                            n_labels=self.config.label_num,
                                            patch_shape = patch_shape,
                                            patch_start_offset = patch_start_offset)


        train_slices = self.get_number_of_patches(data_file, train_list, patch_shape, patch_start_offset)
        validaiton_slices = self.get_number_of_patches(data_file, validation_list, patch_shape, patch_start_offset)

        train_steps = self.get_number_of_steps(train_slices, batch_size)
        validaiton_steps = self.get_number_of_steps(validaiton_slices, batch_size)

        return train_generator, validation_generator, train_steps, validaiton_steps
    
    
    # function signatures
    # def create_patch_index_list(self, indices, image_shape, patch_shape, patch_start_offet = None):
    # def add_data(self, x_list, y_list, data_file, index, patch_shape, augment = False):

    def get_number_of_patches(self, data_file, index_list, patch_shape=None, patch_start_offset=None):
        return len(index_list)

        # if patch_shape:
        #     print("STARTED COUNT OF SLICES")
        #     index_list = self.create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_start_offset)
        #     count = 0
        #     for index in index_list:
        #         print(f"Adding data for slice -{index}")
        #         x_list = list()
        #         y_list = list()
        #         self.add_data(x_list, y_list, data_file, index, patch_shape=patch_shape)
        #         if len(x_list) > 0:
        #             count += 1

        #     print("FINISHED COUNT OF SLICES ")
        #     return count
        # else:
        #     return len(index_list)


    @staticmethod
    def get_number_of_steps(n_samples, batch_size):
        if n_samples <= batch_size:
            return n_samples
        elif np.remainder(n_samples, batch_size) == 0:
            return n_samples//batch_size
        else:
            return n_samples//batch_size + 1



    # @param data_path: path to the hdf file
    def generator(self, data, indices, batch_size = 1, n_labels = 1, patch_shape = None, patch_start_offset = None, augment = False):
        
        old_indices = indices 
        
        while True: 

            x_list = list()
            y_list = list()

            # the last three numbers are the height, width and the depth of the MRI scan
            image_shape = data.root.data.shape[-3:]

            if patch_shape: 
                indices = self.create_patch_index_list(indices, image_shape, patch_shape)
            else:
                indices = old_indices

            shuffle(indices)

            while len(indices) > 0:

                index = indices.pop()

                self.add_data(x_list, y_list, data, index, patch_shape)
                
                if len(x_list) == batch_size or (len(indices) == 0 and len(x_list) > 0):
                    yield self.convert_data(x_list, y_list, n_labels = self.config.label_num, labels = self.config.labels)
                    x_list = list()
                    y_list = list()


    @staticmethod
    def resize(img, shape, mode='constant', orig_shape=(155, 240, 240)):
        """
        Wrapper for scipy.ndimage.zoom suited for MRI images.
        """
        assert len(shape) == 3, "Can not have more than 3 dimensions"
        factors = (
            shape[0]/orig_shape[0],
            shape[1]/orig_shape[1], 
            shape[2]/orig_shape[2]
        )
        
        # Resize to the given shape
        return zoom(img, factors, mode=mode)

    def preprocess(self, img, out_shape=None):
        """
        Preprocess the image.
        Just an example, you can add more preprocessing steps if you wish to.
        """
        if out_shape is not None:
            img = self.resize(img, out_shape, mode='constant')
        
        # Normalize the image
        mean = img.mean()
        std = img.std()
        return (img - mean) / std                    

    def convert_data(self, x_list, y_list, n_labels, labels):

        x = np.asarray(x_list)
        y = np.asarray(y_list)

        print(f"X_SHAPE {x.shape}")
        print(f"Y_SHAPE_BEFORE {y.shape}")

        new_shape = [y.shape[0], n_labels] + list(y.shape[2:])
        new_y = np.zeros(new_shape, np.int8)

        print(f"Y_SHAPE_AFTER {new_y.shape}")

        for label in range(n_labels):
            new_y[:, label][y[0] == labels[label]] = 1

        return x, [x, new_y]



    def add_data(self, x_list, y_list, data_file, index, patch_shape, augment = False):

        data, truth = self.get_data_from_file(data_file, index, patch_shape)

        # ignore augment this time
        # if augment: 
        #     if patch_shape is not None 
        #         affine_matrix = data_file.root.affine[index[0]]
        #     else: 
        #         affine_matrix = data_file.root.affine[index]

        if np.any(truth != 0):
            x_list.append(data)
            y_list.append(truth)


    def get_data_from_file(self, data, index, patch_shape = None):
        
        # image retrieval will done by slices 
        if patch_shape:
            index, slice_index = index 
            data, truth = self.get_data_from_file(data, index)
            x = self.get_patch_from_3d_slice(data, patch_shape, slice_index)
            y = self.get_patch_from_3d_slice(truth, patch_shape, slice_index)
        else: 
            x, y  = data.root.data[index], data.root.true[index]

        return x, y 

    # @param data - the data of some index in the data table 
    @staticmethod
    def get_patch_from_3d_slice(data, patch_shape, slice_index):

        slice_index = np.asarray(slice_index, dtype = np.int16)
        patch_shape = np.asarray(patch_shape)
        image_shape = data.shape[-3:]

        ends = list()

        for index in range(3):
            ends.append(patch_shape[index] + slice_index[index])

        for (index, elem) in enumerate(ends):
            if (elem > image_shape[index]):
                ends[index] = image_shape[index]

        return data[..., slice_index[0]:ends[0],
                         slice_index[1]:ends[1],
                         slice_index[2]:ends[2]]
   



    @staticmethod 
    def compute_patch_indices(image_shape, patch_shape, start):
        if start is None: 
            start = (0,0,0)
        stop = image_shape + start 
        step = patch_shape
        return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)
        

    def create_patch_index_list(self, indices, image_shape, patch_shape, patch_start_offet = None):

        patch_index = list()
        print("*"*31)
        print("INDICES")
        print(indices)
        print("*"*31)

        for index in indices:
            print(f"Creating slices for subject {index}")
            print("-"*31)
            if patch_start_offet is not None:
                start_offset = np.negative(tuple([np.random.choice(patch_start_offet[index] + 1) for index in range(len(patch_start_offet))]))
                patches = self.compute_patch_indices(image_shape, patch_shape, start_offset)
            else:
                patches = self.compute_patch_indices(image_shape, patch_shape, patch_start_offet)
            

            # extend add each input element, treating it seperately
            patch_index.extend(itertools.product([index], patches))

        print("Slices are created for listed subjects")

        return patch_index

