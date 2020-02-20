import numpy as np
import nibabel as nib 
from nilearn.image import crop_img
from scipy.ndimage import zoom
import tables
import os
from model.model import build_model
from utils.config import get_config
from keras.models import load_model

def main():

    path = "/home/alisher/Documents/data/brats/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_10_1"
    input_shape = (4,80,96,64)
    seqs = ["flair","t1","t2","t1ce","seg"]
    data_paths = [{}]
    for name in os.listdir(path):
        for seq in seqs: 
            if seq in name: 
                data_paths[0][seq] = f"{path}/{name}"
                break 

    output_channels = 3

    data = np.empty((len(data_paths[:4]),) + input_shape, dtype=np.float32)
    labels = np.empty((len(data_paths[:4]), output_channels) + input_shape[1:], dtype=np.uint8)

    for i, imgs in enumerate(data_paths):
        try:
            data[i] = np.array([preprocess(nib.load(imgs[m]), input_shape[1:]) for m in seqs], dtype=np.float32)
            labels[i] = preprocess_label(read_img(imgs['seg']), input_shape[1:])[None, ...]
        except Exception as e:
            print(f'Something went wrong with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
            continue
    config, _ = get_config(os.path.abspath('config.json'))

    if os.path.exists(config.model_path):
        print("Pretrained model is loaded")
        model = load_model(config.model_path)
    else:
        print("New model was initialized")
        model = build_model(input_shape = input_shape, output_channels=config.label_num)
    
    model.fit(data, [labels,data],batch_size = 1, epochs = 1)


def read_img(img):
    return nib.load(img).get_data()

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


def preprocess(img, out_shape=None):
    """
    Preprocess the image.
    Just an example, you can add more preprocessing steps if you wish to.
    """
    if out_shape is not None:
        img = resize(img, out_shape, mode='constant')
    
    # Normalize the image
    mean = img.mean()
    std = img.std()
    return (img - mean) / std


def preprocess_label(img, out_shape=None, mode='nearest'):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)
    
    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.uint8)

if __name__ == "__main__":
        main()