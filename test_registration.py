

import os 
import sys
import SimpleITK as sitk 
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from utils.config import get_config
from utils.copyfiles import copy_files_to_dest
import re
from nipype.interfaces.dcm2nii import Dcm2niix
from utils.custom_logger import Logger
import shutil

# global values 
umc_names = [["t2_tirm", "T2_tra_FLAIR"], ["t1_se_sag", "sag_t1"], ["t1_mpr_tra"], ["t2_tse_tra", "T2_tra_2"]]
actual_names = [ "flair", "t1", "t1ce", "t2"]
logger = Logger(path = os.path.abspath('logs/'), name = "preprocessing")

def main():

    test = False


    config, _ = get_config(os.path.abspath('config.json'))

    if test: 

        
        print(sys.argv[0])
        
        image_path = config.data_path
        PARTICULAR_SUBJECT = 10 
        PARTICULAR_FOLDER = "extracted"
        subject_path = f"{image_path}/{PARTICULAR_SUBJECT}/{PARTICULAR_FOLDER}"
        fixed_image_path = config.brats_t1c_sequence_path
        subject_sequences, sequence_names = retrive_subject_sequences(subject_path)

        # the path to BRATS a
        # moving images of a particular subject 
        moving_images = subject_sequences

        for path in moving_images: 
            # check the path validity of each sequence
            assert os.path.exists(path)


        # path to the registration folder
        registered_image_path = f"{config.registration_path}/{PARTICULAR_SUBJECT}"

        if not os.path.exists(registered_image_path):
            os.mkdir(registered_image_path)


    
        '''
            Example of how we applied rigid registration 

            B - BRATS
            s,s1,s2 - shapes of sequences

            Fixed    
            B         T1  -> T1
            s         s1     s

            T1        F   -> F
            s         s2     s



            Just to note: T2 image in Subject 10 is very poor
            
            and so on
        '''


        for (index, moving_image) in enumerate(moving_images): 

            sequence = sequence_names[index]
            output_image = f"{registered_image_path}/{sequence}"
            transform(fixed_image_path, moving_image, output_image)
            transform(fixed_image_path, moving_image, output_image, 2)
            fixed_image_path = output_image
            logger.info(f"Complete registration for {moving_image}")

    else:
        new_path = '/home/alisher/Documents/data/umc/Patology'
        # remove_dirs(new_path)
        # dicom conversion to nii
        #convert_nii(new_path=new_path)
        # skull stripping and rigid registration
        #full_process(fixed_image_path = config.brats_t1c_sequence_path, path = new_path)
        # copy_files_to_dest()

    return


def create_direct(path):
    if not os.path.exists(path):
        os.mkdir(path)

def remove_dirs(path, remove = ['extracted', 'registered_v2']): 
    for folder in os.listdir(path): 
        folder_path = f'{path}/{folder}'
        for child in os.listdir(folder_path):
            if child in remove: 
                shutil.rmtree(f'{folder_path}/{child}')


def convert_nii(path = "/home/alisher/Documents/data/umc/РДЦ/ПАТОЛОГИЯ", new_path = '/home/alisher/Documents/data/umc/Patology', tag = 'converted'):
    
    assert os.path.exists(path)

    create_direct(new_path)

    damaged_files = []

    for folder in os.listdir(path): 
        
        patient_path = f'{path}/{folder}'
        patient_number = os.listdir(patient_path)[0]
        patient_dicom = f'{patient_path}/{patient_number}'

        subject_path = f'{new_path}/{folder}'
        converted_path = f'{subject_path}/{tag}'

        create_direct(subject_path)
        create_direct(converted_path)

        if not len(os.listdir(converted_path)) == 0: 
            logger.info(f"Already converted {folder}")
            continue


        converter = Dcm2niix()
        converter.inputs.compression = 5
        # ignore derived, localizer and 2D images
        converter.inputs.ignore_deriv = True
        converter.inputs.source_dir = patient_dicom
        converter.inputs.output_dir = converted_path
        # run converter
        logger.info(f"Started converting for {folder}")
        try:
            converter.run()
        except: 
            logger.info(f"Something went wrong with {folder}")
            damaged_files.append(folder)
        logger.info(f"Completed converting for {folder}")

    logger.info(f"Finished converting {len(os.listdir(path))} subjects")
    logger.info("*"*30)
    logger.info(f"Damaged_files: {damaged_files}")


def full_process(fixed_image_path, path = "/home/alisher/Documents/data/umc/umc_10_09_benign/processed", extracted_folder = "extracted", converted_folder = "converted", reg_name = "registered_v2"):

    global umc_names, actual_names

    reference_image_path = fixed_image_path

    for folder in os.listdir(path): 

        patient_path = f'{path}/{folder}/{extracted_folder}'

        # path to a subject in the reg_name folder 
        if (not os.path.exists(f'{path}/{folder}/{reg_name}')):
                os.mkdir(f'{path}/{folder}/{reg_name}')
        
        # path to the registration folder
        registered_image_path = f'{path}/{folder}/{reg_name}'

        if not os.path.exists(patient_path): 
            os.mkdir(patient_path)

        logger.info("PATIENT_PATH")
        logger.info(patient_path)

        subject_sequences, sequence_names = retrive_subject_sequences(patient_path, None)

        is_skull_stripped = True

        if (len(sequence_names) == 0 or len(subject_sequences) == 0):
            is_skull_stripped = False


        path_to_old_size = {}

        if not is_skull_stripped: 
            
            converted_path = f'{path}/{folder}/{converted_folder}'
            
            index = 0 
            seq_num = []

            logger.info(f"Patient {folder}")

            for converted in os.listdir(converted_path): 
                for(index, umc_name) in enumerate(umc_names): 
                    for pattern in umc_name:   
                        if "nii.gz" in converted and pattern in converted:  

                            actual_name = actual_names[index]                      
                            skull_stripped_path = f"{patient_path}/{folder}_{actual_name}.nii.gz"

                            if os.path.exists(skull_stripped_path): 
                                logger.info("\n")
                                logger.info(f"Already skull stripped")
                                logger.info(skull_stripped_path)
                                logger.info("*"*30)
                                logger.info("Compairing old size with new size")
                                new_size =  os.stat(f'{converted_path}/{converted}').st_size
                                if new_size < path_to_old_size[skull_stripped_path]:
                                    logger.info("New size is smaller, leave the old version")
                                    continue
                                logger.info("New size is bigger, update old")
                                logger.info("\n")
                                
                            path_to_old_size[skull_stripped_path] = os.stat(f'{converted_path}/{converted}').st_size
                            hd_bet_extraction(f'{converted_path}/{converted}', skull_stripped_path)
                            seq_num.append(pattern)
            
            logger.info(f"Available patient sequences: {seq_num}")
        
        if len(sequence_names) == 0 :
            subject_sequences, sequence_names = retrive_subject_sequences(patient_path, None)
        
        for (index, moving_image) in enumerate(subject_sequences):
            
            sequence = sequence_names[index]
            output_image = f"{registered_image_path}/{sequence}"
            
            if os.path.exists(output_image): 
                logger.info("Already registered")
                logger.info(output_image)
                logger.info("*"*30)
                continue

            transform(reference_image_path, moving_image, output_image)
            transform(reference_image_path, moving_image, output_image, 2)
            reference_image_path = output_image
            logger.info(f"Complete registration for {moving_image}")

        reference_image_path = fixed_image_path
    
    return 




# by defualt hdbet is used
def hd_bet_extraction(image_path, skull_stripped_path):
    
    try:
        os.system(f"hd-bet -i {image_path} -o {skull_stripped_path}")
        logger.info(f"STRIPPED SUCCESFULLY for {image_path}")
    except: 
        assert Exception(f"Not possible to skull strip this subject, check the file {image_path}")


'''
    Method which applies a particular registration method and saves the result image to output_image_path
    @param option - the type of transofrmation, 0 - ridig, 1 - non rigidd
'''

def transform(fixed_image_path, moving_image_path, output_image_path, option = 0):
    
    fixed_image =  sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32) 
    
    if option == 0: 
       
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                    moving_image, 
                                                    sitk.Euler3DTransform(), 
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)

        registration_method = sitk.ImageRegistrationMethod()

        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        registration_method.SetInterpolator(sitk.sitkLinear)

        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100) #, estimateLearningRate=registration_method.EachIteration)
        registration_method.SetOptimizerScalesFromPhysicalShift() 

        final_transform = sitk.Euler3DTransform(initial_transform)
        registration_method.SetInitialTransform(final_transform)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                    sitk.Cast(moving_image, sitk.sitkFloat32))

        save_image(final_transform, fixed_image, moving_image, output_image_path)

    elif option == 1: 

        """
            Non-rigid registration, in development
        """
        
        pass
                             
    elif option == 2:
        # checking the final sizes
        n_fixed = nib.load(fixed_image_path)
        n_moved = nib.load(moving_image_path)
        n_registered = nib.load(output_image_path)

        logger.info(f"Fixed shape : {n_fixed.shape}, Moved Shape {n_moved.shape}, Registered Shape{n_registered.shape}")
    

def retrive_subject_sequences(path, sequence_to_ignore = "t2"):

    path_sequences = []
    sequences = []

    for (_, sequence) in enumerate(os.listdir(path)):
        if ("mask" in sequence) or (not sequence_to_ignore is None and sequence_to_ignore in sequence):
            continue
        path_sequences.append(f"{path}/{sequence}")
        sequences.append(sequence)

    return path_sequences,sequences

    

def save_image(final_transform, fixed_image, moving_image, outputfile_prefix):

    resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    # [
    #     "/home/alisher/Documents/data/umc/sep10/hd-bet-resampling/10/extracted/10_flair.nii.gz", 
    #     "/home/alisher/Documents/data/umc/sep10/hd-bet-resampling/10/extracted/10_t1.nii.gz", 

    # ]
    sitk.WriteImage(resampled, outputfile_prefix)


def show_slices(slices):
    """ Function to display row of image slices """
    
    fig, axes = plt.subplots(1, len(slices))
    
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


    
if __name__ == "__main__":
    main()



