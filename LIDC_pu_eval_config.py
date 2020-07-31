import os
from loaders import *
import glob




#########################################
#             evaluation 			    #
#########################################

num_samples = 16
patch_size = [180, 180]

out_dir = '/home/syl/Documents/Project/prob_reg/images/test_result/LIDC_pu' # saved sample and eval directory
os.makedirs(out_dir, exist_ok=True)

data_path_base = '/home/syl/Documents/Project/prob_reg/dataset/hpu-lung/'
data_loader = LIDC_IDRI(path_base = data_path_base, list_id='test')

result_path = '/home/syl/Documents/Project/prob_reg/download/images/test_LIDC_pu_07_18_2020_14_31/'
file_list = glob.glob(result_path+"*prob.npy")
# print(len(file_list[1000:]))

