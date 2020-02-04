
from utils.config import get_config
import os
from data_helper import DataHelper
from utils.custom_logger import Logger
import pandas as pd



def main(): 

    logger = Logger(path = os.path.abspath('logs/'), name = "mri_loss_log")
    config, _ = get_config(os.path.abspath('config.json'))
    data_helper = DataHelper(config, logger)  
    missing_dict, history = data_helper.check_missing()  
    s = pd.Series(missing_dict, name = "Modalities")
    s.to_csv(os.path.abspath("missing_two.csv"))


    for (index, id) in enumerate(history.keys()): 
        # logger.info(index)
        # logger.info(f"{id} - {history[id]}")
        if (len(history[id]) > 1):
            logger.info(f"{id} - {history[id]}")


if __name__ == "__main__":
    main()  



