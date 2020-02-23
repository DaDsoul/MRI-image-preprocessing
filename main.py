
from utils.config import get_config
import os
from data_helper import DataHelper
from utils.custom_logger import Logger
import pandas as pd
from model.model import build_model
from test_registration import get_mri_sequence
from data_loader.data import DataGenerator
from model.trainer import Trainer
import tables
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from utils.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)
from functools import partial
import math
from model.model_unet import unet_model_3d
from model.new_trainer import DataGeneratorNew

CHANNELS_NUM = 4

def main(): 


    # preprocessing step 
    logger = Logger(path = os.path.abspath('logs/'), name = "trainining_process")
    config, _ = get_config(os.path.abspath('config.json'))
    # data_helper = DataHelper(config, logger)  
    # missing_dict, history = data_helper.check_missing()  
    # s = pd.Series(missing_dict, name = "Modalities")
    # s.to_csv(os.path.abspath("missing_two.csv"))


    # preprocessing step 
    # for (index, id) in enumerate(history.keys()): 
    #     if (len(history[id]) > 1):
    #         logger.info(f"{id} - {history[id]}")

    
    mri_image = get_mri_sequence(config.brats_flair_sequence_path)
    input_shape = (CHANNELS_NUM,) + tuple(config.image_shape)
    
    # print(mri_image.get_data())

    data_generator = DataGenerator(config)
    main_path = config.main_path
  
    hdf_path = f"{main_path}/data/{config.data_set}"
    data_file = tables.open_file(hdf_path, 'r')

    train_indices, validaiton_indices = data_generator.validation_split(f"{main_path}/data/train_keys.csv", f"{main_path}/data/validation_kes.csv")
    train_indices = list(train_indices)
    validaiton_indices = list(validaiton_indices)
    trainer = Trainer(config)

    # logger.info("Generators are creating")
    # train_generator, validation_generator, train_steps, validation_steps = trainer.get_generators(data_file, train_indices,validaiton_indices, config.patch_shape, config.patch_start_offset, config.batch_size)
    # logger.info("Generators are created")

    params = {
        'n_labels':config.label_num,
        'labels':config.labels,
        'dim':tuple(config.image_shape), 
        'batch_size':1, 
        'shuffle':True
    }

    train_generator = DataGeneratorNew(train_indices, data_file, **params)
    validation_generator = DataGeneratorNew(validaiton_indices, data_file, **params)

    logger.info("Model is loading")
    if os.path.exists(config.model_path):
        logger.info("Pretrained model is loaded")
        model = load_model(config.model_path)
    else:
        logger.info("New model was initialized")
        model = build_model(input_shape = input_shape, output_channels=config.label_num)

    logger.info("Training model is started")

    model.fit_generator(generator=train_generator,
                        epochs=config.epochs,
                        validation_data=validation_generator)
    



def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))



if __name__ == "__main__":
    main()  



