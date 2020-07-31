
import numpy as np
import torch, torchvision
from torchvision.utils import save_image
import glob
import PIL.Image as Image
import os
from PIL.Image import open as imread

def save_combined(file_path, num_image_in_row, num_image_in_col):
    os.makedirs(file_path + '/combined/', exist_ok=True)
    file_list = glob.glob(file_path+"*sample.png")
    # print(file_list)
    step = num_image_in_row * num_image_in_col
    for i in range(0, len(file_list), step):
        # print(i)
        tmp = []
        for j in range(step):
            if i+j < len(file_list):
                print(i + j)
                img = np.array(imread(file_list[i+j]))[None, ..., 0, None] / 255.
                img = img.transpose(0, 3, 1, 2)
                # print(img.shape)
                tmp.append(img)
        # print(tmp)
        tmp = np.concatenate(tmp, axis=0)
        print('saving ...')
        tmp = torch.from_numpy(tmp).float()
        # print(tmp.shape)
        save_image(tmp, file_path + 'combined/{}_{}.png'.format(i, i+j), num_image_in_row, pad_value=1)
        print('saved')


if __name__ == '__main__':
    # save_combined('/home/syl/Documents/Project/prob_reg/images/Toshow/MixMNIST_supp/', \
                #    5, 5)
    # save_combined('/home/syl/Documents/Project/prob_reg/images/Toshow/MixMNIST_pu_supp/', \
    #                5, 5)
    save_combined('/home/syl/Documents/Project/prob_reg/images/Toshow/LIDC/', \
                   2, 4)
    # save_combined('/home/syl/Documents/Project/prob_reg/images/Toshow/LIDC_pu/', \
                #    2, 4)


