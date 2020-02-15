
from utils.config import get_config
import os
from data_helper import DataHelper
from utils.custom_logger import Logger
import pandas as pd
# from model.model import build_model
from test_registration import get_mri_sequence
from data_loader.data import DataGenerator
from model.trainer import Trainer


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

    
    # mri_image = get_mri_sequence(config.brats_flair_sequence_path)
    # input_shape = (CHANNELS_NUM,) + mri_image.shape
    
    # print(mri_image.get_data())

    data_generator = DataGenerator(config)

    main_path = config.main_path

    trainer = Trainer(config)

    indices = trainer.create_patch_index_list([3,4,5], config.image_shape, [10,10,10])
    
    
    #:param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.

    for index in indices: 
        print(index)
        print("*"*30)
    # train, test = data_generator.validation_split(2, f"{main_path}/data/train_keys.csv", f"{main_path}/data/validation_kes.csv")
    # print(train.shape)
    # print(test.shape)

    # mri_model = build_model(input_shape, CHANNELS_NUM)




if __name__ == "__main__":
    main()  



