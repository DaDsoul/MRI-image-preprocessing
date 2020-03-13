import tables
import numpy as npi
import nibabel as nib
import numpy as np
import os
from random import shuffle
import pandas as pd
from nilearn.image import crop_img
from scipy.ndimage import zoom




class DataGenerator: 


    # signature fill_data(self, image_files, out_file, image_shape, samples_num, channels =4, truth_dtype=np.uint8, subject_ids=None, normalize=True):

    def __init__(self, config):
        self.config = config
        hgg_data = self.config.brats_data_path_training_hgg
        lgg_data = self.config.brats_data_path_training_lgg
        
        self.data_set_path = f"{self.config.main_path}/data/{self.config.data_set}"
        samples_num = len(os.listdir(hgg_data)) + len(os.listdir(lgg_data))
        
        if (os.path.exists( self.data_set_path)):
            print("Data set file already exists")
        else:
            print("Data filling is started")
            self.fill_data(hgg_data,  self.data_set_path, self.config.image_shape, samples_num)
        

    def write_image_data_to_file(self, image_files, data_storage, truth_storage, truth_dtype, channels = 4):
        for set_of_files in os.listdir(image_files):
            images = self.read_images(f"{image_files}/{set_of_files}")
            subject_data = [image for image in images]
            print("-"*30)
            print(f"Adding {set_of_files}")
            self.add_data_to_storage(data_storage, truth_storage, subject_data, channels,
                                truth_dtype)
            print(f"Added {set_of_files}")
            print("-"*30)

        
        print("Writing images is finished")
        return data_storage, truth_storage

    # check data and true group in the hdf_file
    def check_data(self):
        data_file = tables.open_file(self.data_set_path, mode = 'r')
        # check overall shape
        print(f"Data shape {data_file.root.true.shape}")    
        print(f"Ground Truth shape {data_file.root.data.shape}")
        first = data_file.root.data[1]
        mean = first.mean(axis = (1,2,3))
        print(f"mean for some subject {mean}")
        print(f"shape for some subject {first.shape}")
        some_sequence = first[0]
        new_image = nib.Nifti1Image(some_sequence, affine = np.eye(4))
        nib.save(new_image, f"/home/alisher/Desktop/first.nii.gz")
        data_file.close()
    

    # param images_files - path to the mri image folder
    def read_images(self, path_to_subject, new_shape = [144,144,144], normalize = False):
        crop = self.config.is_compress == 1
        index_truth = -1
        subject_sequences = list()  

        paths = os.listdir(path_to_subject)

        for index, sequence in enumerate(paths):
            if "seg" in sequence: 
                index_truth = index
            
            image = nib.load(f"{path_to_subject}/{sequence}")
            image_matrix = image.get_data()
            

            if crop: 
                orig_shape = image_matrix.shape

                factors = ( new_shape[0]/orig_shape[0],
                            new_shape[1]/orig_shape[1], 
                            new_shape[2]/orig_shape[2] )

                image_matrix = zoom(image_matrix, factors,  mode = "constant")

                # image normalization 
                if not index_truth == index and normalize:
                    mean = image_matrix.mean()
                    std = image_matrix.std() 
                    print("Done with the normalization")
                    image_matrix = (image_matrix - mean)/std

                
            subject_sequences.append(image_matrix)
        
        
        last_index = len(paths) - 1

        if not (index_truth == last_index):
            temp = subject_sequences[index_truth]
            subject_sequences[index_truth] = subject_sequences[last_index]
            subject_sequences[last_index] = temp

        return subject_sequences
    
    
    def normalize_data_storage(self, data_storage):
        means = list()
        stds = list()

        for index in range(data_storage.shape[0]):
            data = data_storage[index]
            means.append(data.mean(axis=(1, 2, 3)))
            stds.append(data.std(axis=(1, 2, 3)))
        
        mean = np.asarray(means).mean(axis=0)
        std = np.asarray(stds).mean(axis=0)
        
        for index in range(data_storage.shape[0]):
            data_storage[index] = self.normalize_data(data_storage[index], mean, std)
            
        return data_storage


    def fill_data(self, image_files, out_file, image_shape, samples_num, channels =4, truth_dtype=np.uint8, subject_ids=None, normalize=False):

        
        try: 
            hdf_file, data_storage, truth_storage, _ = self.create_data_file(out_file, n_samples = samples_num, image_shape = image_shape)
        except Exception as e: 
            os.remove(out_file)
            raise e

        self.write_image_data_to_file(image_files, data_storage, truth_storage, truth_dtype, channels)

        if normalize:
            self.normalize_data_storage(data_storage)
        hdf_file.close()

        return out_file

    @staticmethod
    def add_data_to_storage(data_storage, truth_storage, subject_data, channels, truth_dtype):
        data_storage.append(np.asarray(subject_data[:channels])[np.newaxis])
        
        # with dtype
        # truth_storage.append(np.asarray(subject_data[channels], dtype=truth_dtype)[np.newaxis][np.newaxis])

        # without dtype 
        truth_storage.append(np.asarray(subject_data[channels])[np.newaxis][np.newaxis])



        # affine_storage.append(np.asarray(affine)[np.newaxis])


    # create pytable 
    @staticmethod
    def create_data_file(out_file, n_samples, image_shape, channels = 4):
        hdf5_file = tables.open_file(out_file, mode='w')

        # complevel - compression level 
        # complib - the library for compression
        filters = tables.Filters(complevel=5, complib='blosc')
        
        data_shape = tuple([0, channels] + list(image_shape))
        
        truth_shape = tuple([0, 1] + list(image_shape))
        
        data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                            filters=filters, expectedrows=n_samples)
        
        truth_storage = hdf5_file.create_earray(hdf5_file.root, 'true', tables.UInt8Atom(), shape=truth_shape,
                                                filters=filters, expectedrows=n_samples)
        
        affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
        return hdf5_file, data_storage, truth_storage, affine_storage


    @staticmethod
    def normalize_data(data, mean, std):
        data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
        data /= std[:, np.newaxis, np.newaxis, np.newaxis]
        return data


    #splits the data into the validation and training sets, if already exists returns the indices 
    @staticmethod
    def validation_split(train_path_keys, validation_path_keys, data_file = None, data_split = 0.8):

        if not os.path.exists(train_path_keys):
            print("Creating indexes for training and validation sets")
            if data_split is None: 
                assert Exception("The reference to the data is needed")
            indices = list(range(data_file.root.data.shape[0]))
            shuffle(indices)
            training_end_index = int(len(indices)*data_split)
            for_training_indices = indices[:training_end_index]
            for_validation_indices = indices[training_end_index:]
            pd.DataFrame({"indices":for_training_indices}).to_csv(train_path_keys)
            pd.DataFrame({"indices":for_validation_indices}).to_csv(validation_path_keys)
            return for_training_indices, for_validation_indices
        else:
            return pd.read_csv(train_path_keys)["indices"].to_numpy(), pd.read_csv(validation_path_keys)["indices"].to_numpy()
            
