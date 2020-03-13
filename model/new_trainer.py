import numpy as np
import keras
import itertools
import random

class DataGeneratorNew(keras.utils.Sequence):

    def __init__(self, list_IDs, data_file, n_labels, labels, image_shape = (144,144,144),patch_shape = None, patch_start_offset = (0,0,0),batch_size=32, dim=(32,32,32), shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels = labels
        self.shuffle = shuffle
        self.data_file = data_file 
        self.n_labels = n_labels
        self.patch_shape = patch_shape
        self.patch_start_offset = patch_start_offset
        self.image_shape = image_shape
        self.on_epoch_end()


    def __len__(self):
        if not self.patch_shape:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        else: 
            # count = self.get_number_of_patches(self.data_file, self.list_IDs, self.patch_shape, self.patch_start_offset) 
            count = len(self.indexes)
            return int(count / self.batch_size)


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if not self.patch_shape:
            self.indexes = np.arange(len(self.list_IDs))
        else: 
            all_indexes = self.create_patch_index_list(self.list_IDs, self.image_shape, self.patch_shape)
            self.indexes = []
            batch = 100

            yes = False  

            if not yes: 
                
                for (index, candidate) in enumerate(all_indexes): 
                    _, truth = self.get_data_from_file(self.data_file, candidate, self.patch_shape)
                    if (np.any(truth != 0)):
                        self.indexes.append(candidate)
                        if (index % batch == 0):
                            print(f"Len of valid slices for batch {index/batch} is {len(self.indexes)}")
                
                print(f"Final Len of valid slices {len(self.indexes)}")
            else: 
                self.indexes = all_indexes

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def get_number_of_patches(self, data_file, index_list, patch_shape=None, patch_start_offset=None):

        if patch_shape:
            index_list = self.create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_start_offset)
            count = 0
            for index in index_list:
                x_list = list()
                y_list = list()
                self.add_data(x_list, y_list, data_file, index, patch_shape=patch_shape)
                if len(x_list) > 0:
                    count += 1

            return count
        else:
            return len(index_list)

    @staticmethod 
    def compute_patch_indices(image_shape, patch_shape, start):
        if start is None: 
            start = (0,0,0)
        stop = image_shape + start 
        step = patch_shape
        return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)
        

    def create_patch_index_list(self, indices, image_shape, patch_shape = None, patch_start_offet = None):

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


    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list() 
        y_list = list() 
        

        self.add_data(x_list, y_list, self.data_file, indexes, patch_shape= self.patch_shape)
        
        x = np.asarray(x_list)
        y = np.asarray(y_list)
        
        new_shape = [y.shape[0], self.n_labels] + list(y.shape[2:])
        new_y = np.zeros(new_shape, np.int8)

        for label in range(self.n_labels):
            new_y[:, label][y[0] == self.labels[label]] = 1

        return x, new_y

    
    def get_data_from_file(self, data, index, patch_shape = None):
        # image retrieval will done by slices 
        if patch_shape:
            index, slice_index = index 
            data, truth = self.get_data_from_file(data, index)
            x = self.get_patch_from_3d_data(data, patch_shape, slice_index)
            y = self.get_patch_from_3d_data(truth, patch_shape, slice_index)
        else: 
            x, y  = data.root.data[index], data.root.true[index]

        return x, y 

    def add_data(self, x_list, y_list, data_file, index, patch_shape = None, augment = False, permute = True):

        data, truth = self.get_data_from_file(data_file, index, patch_shape)


        print(data.shape)
        print(truth.shape)
        
        if patch_shape and permute: 
            data, truth = self.random_permutation_x_y(data, truth[np.newaxis])


        x_list.append(data)
        y_list.append(truth)
        
        # we are adding non-labeled data as well for model robustness
        # if np.any(truth != 0):
            
        #     print("I AM HERE")

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
   
    def get_patch_from_3d_data(self, data, patch_shape, patch_index):
        """
        Returns a patch from a numpy array.
        :param data: numpy array from which to get the patch.
        :param patch_shape: shape/size of the patch.
        :param patch_index: corner index of the patch.
        :return: numpy array take from the data with the patch shape specified.
        """
        patch_index = np.asarray(patch_index, dtype=np.int16)
        patch_shape = np.asarray(patch_shape)
        image_shape = data.shape[-3:]
        if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
            data, patch_index = self.fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
        
        return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                    patch_index[2]:patch_index[2]+patch_shape[2]]

    @staticmethod
    def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
        """
        Pads the data and alters the patch index so that a patch will be correct.
        :param data:
        :param patch_shape:
        :param patch_index:
        :return: padded data, fixed patch index
        """
        image_shape = data.shape[-ndim:]
        pad_before = np.abs((patch_index < 0) * patch_index)
        pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
        pad_args = np.stack([pad_before, pad_after], axis=1)
        if pad_args.shape[0] < len(data.shape):
            pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
        data = np.pad(data, pad_args, mode="edge")
        patch_index += pad_before
        return data, patch_index

    def random_permutation_x_y(self, x_data, y_data):
        """
        Performs random permutation on the data.
        :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
        :param y_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
        :return: the permuted data
        """
        key = self.random_permutation_key()

        return self.permute_data(x_data, key), self.permute_data(y_data, key)

    @staticmethod
    def generate_permutation_keys():
        return set(itertools.product(itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2), range(2)))

    def random_permutation_key(self):
        """
        Generates and randomly selects a permutation key. See the documentation for the
        "generate_permutation_keys" function.
        """
        return random.choice(list(self.generate_permutation_keys()))

    @staticmethod
    def permute_data(data, key):
        """
        Permutes the given data according to the specification of the given key. Input data
        must be of shape (n_modalities, x, y, z).
        Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)
        As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
        rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
        transposed.
        """
        data = np.copy(data)
        (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

        if rotate_y != 0:
            data = np.rot90(data, rotate_y, axes=(1, 3))
        if rotate_z != 0:
            data = np.rot90(data, rotate_z, axes=(2, 3))
        if flip_x:
            data = data[:, ::-1]
        if flip_y:
            data = data[:, :, ::-1]
        if flip_z:
            data = data[:, :, :, ::-1]
        # if transpose:
        #     for i in range(data.shape[0]):
        #         data[i] = data[i].T
        return data