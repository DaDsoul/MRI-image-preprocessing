import numpy as np
import nibabel as nib 
from nilearn.image import crop_img
from scipy.ndimage import zoom



def main():

    # path = "/home/alisher/Documents/data/brats/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_flair.nii.gz"
    # image = nib.load(path)    
    # shape = [160,192,128]
    # orig_shape = image.shape

    # print(image.shape)
    # factors = (
    #     shape[0]/orig_shape[0],
    #     shape[1]/orig_shape[1], 
    #     shape[2]/orig_shape[2]
    # )

    # print(zoom(image.get_data(),factors, mode = "constant").shape)

    pass
    

'''
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
'''

def resize(img, orig_shape, shape = (155, 240, 240), mode='constant'):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """


    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        orig_shape[0]/shape[0],
        orig_shape[1]/shape[1], 
        orig_shape[2]/shape[2]
    )

    print(factors)
    
    # Resize to the given shape
    return zoom(img, factors, mode=mode)



def preprocess_label(img, new_shape = [80, 96, 64], mode='nearest'):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)
    
    ncr = resize(ncr, new_shape, mode=mode)
    ed = resize(ed, new_shape, mode=mode)
    et = resize(et, new_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.uint8)


def some_function(data):

    for data_point in data: 
        
        yield kk(data_point)

def kk(elem):
    print("Operation")
    return elem + 2



def compute_patch_indices(image_shape, patch_size, overlap, start=None):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
    if start is None:
        n_patches = np.ceil(image_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow/2)
        print(start, image_shape, patch_size)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_size - overlap
    return get_set_of_patch_indices(start, stop, step)

def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)




if __name__ == "__main__":
        main()