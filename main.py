
from utils.config import get_config
import os
from data_helper import DataHelper
from utils.custom_logger import Logger



def main(): 

    logger = Logger(path = os.path.abspath('logs/'), name = "mri_loss_log")
    config, _ = get_config(os.path.abspath('config.json'))
    data_helper = DataHelper(config, logger)  
    missing_dict = data_helper.check_missing()  

    for key in missing_dict.keys():
        print(missing_dict[key])


if __name__ == "__main__":
    main()



