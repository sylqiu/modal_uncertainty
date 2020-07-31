import os
from loaders import *
import glob




#########################################
#             evaluation 			    #
#########################################

num_samples = 16
patch_size = [180, 180]

out_dir = '/home/syl/Documents/Project/prob_reg/images/test_result/CityScape/' # saved sample and eval directory
os.makedirs(out_dir, exist_ok=True)

data_path_base = '/home/syl/Documents/Project/prob_reg/dataset/cityscape/'
data_loader = CityScapeSwitch(path_base = data_path_base, list_id='val')

result_path = '/home/syl/Documents/Project/prob_reg/download/images/test_LIDC_07_18_2020_14_27/'
file_list = glob.glob(result_path+"*prob.npy")
# print(len(file_list[1000:]))

num_modes = 32

ignore_label = 255