import numpy as np
import keras

class DataGeneratorNew(keras.utils.Sequence):

    def __init__(self, list_IDs, data_file, n_labels, labels, batch_size=32, dim=(32,32,32), shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels = labels
        self.shuffle = shuffle
        self.data_file = data_file 
        self.n_labels = n_labels
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list() 
        y_list = list() 
        self.add_data(x_list, y_list, self.data_file, indexes, False)

        x = np.asarray(x_list)
        y = np.asarray(y_list)

        new_shape = [y.shape[0], self.n_labels] + list(y.shape[2:])
        new_y = np.zeros(new_shape, np.int8)

        for label in range(self.n_labels):
            new_y[:, label][y[0] == self.labels[label]] = 1

        return x, [new_y,x]

    
    def get_data_from_file(self, data, index, patch_shape = None):
        
        # image retrieval will done by slices 
        
        x, y  = data.root.data[index], data.root.true[index]

        return x, y 

    def add_data(self, x_list, y_list, data_file, index, patch_shape, augment = False):

        data, truth = self.get_data_from_file(data_file, index, patch_shape)

        if np.any(truth != 0):
            x_list.append(data)
            y_list.append(truth)

