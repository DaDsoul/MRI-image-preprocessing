

import os 
import sys
import SimpleITK as sitk 
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
moving_image = 0
iteration_number = 0

def main():
    print(sys.argv[0])

    fixed_image_path = "/home/alisher/Documents/data/brats/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_flair.nii.gz"

    moving_images = [
        "/home/alisher/Documents/data/umc/sep10/hd-bet-resampling/10/extracted/10_flair.nii.gz", 
        "/home/alisher/Documents/data/umc/sep10/hd-bet-resampling/10/extracted/10_t1.nii.gz"
    ]

    # fixed_image_path = moving_images[0]


    outputImageFile = "/home/alisher/Documents/data/umc/test/output.nii.gz"
    outputImageFileTwo = "/home/alisher/Documents/data/umc/test/output_two.nii.gz"
    option = 0

    if option == 0:

        fixed_image =  sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(outputImageFile, sitk.sitkFloat32) 

        initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                    moving_image, 
                                                    sitk.Euler3DTransform(), 
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)

        #multi-resolution rigid registration using Mutual Information
        # registration_method = sitk.ImageRegistrationMethod()
        # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        # registration_method.SetMetricSamplingPercentage(0.01)
        # registration_method.SetInterpolator(sitk.sitkLinear)
        # registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
        #                                                 numberOfIterations=100, 
        #                                                 convergenceMinimumValue=1e-6, 
        #                                                 convergenceWindowSize=10)
        # registration_method.SetOptimizerScalesFromPhysicalShift()
        # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        # registration_method.SetInitialTransform(transform)
        # final_transform = registration_method.Execute(fixed_image, moving_image)
        # moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        # sitk.WriteImage(moving_resampled,outputImageFile)

        # sitk.WriteTransform(transform, outputImageFile)


        # rigid registration
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

        save_image(final_transform, fixed_image, moving_image, outputImageFileTwo)

    elif option == 1:

        fixed_image =  sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(outputImageFile, sitk.sitkFloat32) 

        # Select a Demons filter and configure it.
        demons_filter =  sitk.DiffeomorphicDemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(20)
        # Regularization (update field - viscous, total field - elastic).
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(0.8)

        # create initial transform
        initial_tfm = sitk.CenteredTransformInitializer(fixed_image, 
                                                    moving_image, 
                                                    sitk.Euler3DTransform(), 
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
# Run the registration.
        final_tfm = multiscale_demons(registration_algorithm=demons_filter, 
                                    fixed_image = fixed_image,
                                    moving_image = moving_image,
                                    initial_transform = initial_tfm,
                                    shrink_factors = [6,4,2],
                                    smoothing_sigmas = [6,4,2])

        save_image(final_tfm, fixed_image, moving_image, outputImageFileTwo)
                             
    elif option == 2:
        # checking the final sizes
        n_fixed = nib.load(fixed_image_path)
        n_moved = nib.load(moving_images[0])
        n_registered = nib.load(outputImageFile)

        print(n_fixed.shape, n_moved.shape, n_registered.shape)


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

        # Select a Demons filter and configure it.
        demons_filter =  sitk.DiffeomorphicDemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(20)
        # Regularization (update field - viscous, total field - elastic).
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(0.8)

        # create initial transform
        initial_tfm = sitk.CenteredTransformInitializer(fixed_image, 
                                                    moving_image, 
                                                    sitk.Euler3DTransform(), 
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
        # Run the registration.
        final_tfm = multiscale_demons(registration_algorithm=demons_filter, 
                                    fixed_image = fixed_image,
                                    moving_image = moving_image,
                                    initial_transform = initial_tfm,
                                    shrink_factors = [6,4,2],
                                    smoothing_sigmas = [6,4,2])

        save_image(final_tfm, fixed_image, moving_image, output_image_path)
                             
    elif option == 2:
        # checking the final sizes
        n_fixed = nib.load(fixed_image_path)
        n_moved = nib.load(moving_image_path)
        n_registered = nib.load(output_image_path)

        print(n_fixed.shape, n_moved.shape, n_registered.shape)





def save_image(transform, fixed_image, moving_image, outputfile_prefix):
                    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.     
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(transform)
    sitk.WriteImage(resample.Execute(moving_image), outputfile_prefix)

def demons_registration(fixed_image, moving_image, fixed_points = None, moving_points = None):
    
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))
    
    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0) 
    
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(10) #intensities are equal if the difference is less than 10HU
        
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])    

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer    
    #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)        
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
        
    return registration_method.Execute(fixed_image, moving_image)    


def show_slices(slices):
    """ Function to display row of image slices """
    
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform = None, 
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors (list of lists or scalars): Shrink factors relative to the original image's size. When the list entry, 
                                                   shrink_factors[i], is a scalar the same factor is applied to all axes.
                                                   When the list entry is a list, shrink_factors[i][j] is applied to axis j.
                                                   This allows us to specify different shrink factors per axis. This is useful
                                                   in the context of microscopy images where it is not uncommon to have
                                                   unbalanced sampling such as a 512x512x8 image. In this case we would only want to 
                                                   sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
        smoothing_sigmas (list of lists or scalars): Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns: 
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))
    
    # Create initial displacement field at lowest resolution. 
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform, 
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(), 
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])
     fixed_image_path = "/home/alisher/Documents/data/brats/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_flair.nii.gz"
    output = "/home/alisher/Documents/data/umc/test/output.nii.gz"
    n_registered = nib.load(fixed_image_path).get_fdata()
    slice_0 = n_registered[26, :, :]
    slice_1 = n_registered[:, 30, :]
    slice_2 = n_registered[:, :, 16]
    print(slice_0)
    show_slices([slice_0, slice_1, slice_2])
    # Run the registration.            
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1], 
                                                                moving_images[-1], 
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.    
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
            initial_displacement_field = sitk.Resample (initial_displacement_field, f_image)
            initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)
def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
    """
    Args:
        image: The image we want to resample.
        shrink_factor(s): Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma(s): Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
    """
    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors]*image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas]*image.GetDimension()

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)
    
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(shrink_factors,original_size)]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(), 
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0, 
                         image.GetPixelID())

    
if __name__ == "__main__":
    main()



