import tables
import numpy as np

def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')


    # complevel - compression level 
    # complib - the library for compression
    filters = tables.Filters(complevel=5, complib='blosc')
    
    data_shape = tuple([0, n_channels] + list(image_shape))
    
    truth_shape = tuple([0, 1] + list(image_shape))
    
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'true_label', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    
    return hdf5_file, data_storage, truth_storage





