
import pandas 
from collections import defaultdict
import os 
import glob



class DataHelper:


    def __init__(self, config, logger):
        
        self.logger = logger
        self.config = config
        self.name_mapper = defaultdict(lambda: "Invalid")
        self.map_initialize()

    
    def map_initialize(self):
        
        for (index, key) in enumerate(self.config.umc_names): 
            self.name_mapper[key] = self.config.actual_names[index]

 
    def check_missing(self):
        
        missing_count = defaultdict(lambda : 0)
        path = self.config.data_path

        for elem in os.listdir(path):
            folder_path = os.path.abspath(f"{path}/{elem}/{self.config.target_folder}")
            
            for name in self.config.umc_names:
                files_with_name = glob.glob(os.path.join(folder_path, f"*{name}*.nii.gz"))
                if len(files_with_name) == 0:
                    actual_name = self.name_mapper[name] 
                    self.logger.info(f"Missing modality {actual_name} in {elem}")
                    missing_count[actual_name] += 1
        
        








