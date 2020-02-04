
import pandas 
from collections import defaultdict
import os 
import glob
import pydicom


class DataHelper:


    def __init__(self, config, logger):
        
        self.logger = logger
        self.config = config
        self.name_mapper = defaultdict(lambda: "Invalid")
        self.index_mapper = defaultdict(lambda: -1)
        self.map_initialize()

    
    def map_initialize(self):

        for(index, key) in enumerate(self.config.actual_names):
            self.index_mapper[key] = index

        for (index, key) in enumerate(self.config.umc_names): 
            self.name_mapper[key] = self.config.actual_names[index]
            
 
    def check_missing(self):
        
        missing_count = defaultdict(lambda : [0,0,0,0, ""])
        history = defaultdict(lambda : [])
        path = self.config.data_path
        processed_data_path = self.config.processed_data_path

        for elem in os.listdir(path):
            folder_path = os.path.abspath(f"{path}/{elem}/{self.config.target_folder}")
            
            # self.logger.info(f"Inside the {elem} folder")
            for name in self.config.umc_names:
                files_with_name = glob.glob(os.path.join(folder_path, f"*{name}*.nii.gz"))
                actual_name = self.name_mapper[name] 
                if len(files_with_name) == 0:
                    self.logger.info(f"Missing modality {actual_name}")
                else: 
                    missing_count[elem][self.index_mapper[actual_name]] = 1
                    self.logger.info(f"Existing modality {actual_name}")

        for elem in os.listdir(processed_data_path):
            folder_path = os.path.abspath(f"{processed_data_path}/{elem}")
            random_25_th = os.listdir(folder_path)[25]
            path_to_dicom = f"{folder_path}/{random_25_th}"
            patientId = pydicom.dcmread(path_to_dicom).PatientID
            missing_count[elem][4] = patientId
            history[patientId].append(missing_count[elem]) 

        return missing_count, history








