import os
import shutil  


def copy_files_to_dest(folder = '/home/alisher/Documents/data/umc/Patology', folder_to_copy = 'registered_v2',dest_folder = '/home/alisher/Documents/data/umc/Patology_only_register'):
    dir_files = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    for folder in dir_files:
        q =([x[0] for x in os.walk(folder)])
        for q2 in q:
            last = q2.rsplit('/', 1)[-1]
            if last == folder_to_copy:
                dest = (str(dest_folder)+'/'+str(q2.rsplit('/', 2)[-2])+'/'+str(q2.rsplit('/', 2)[-1]))
                shutil.copytree(q2, dest, copy_function = shutil.copy)  
