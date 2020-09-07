import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle
from os.path import join as pjoin
import PIL.Image as Image
from PIL.Image import open as imread
from PIL.Image import fromarray
import torchvision.transforms.functional as TF
from collections import OrderedDict,defaultdict
from cityscapes_labels import labels as cs_labels_tuple
import matplotlib.pyplot as plt
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# random.seed(0)

        
class KittiMonoDepth(Dataset):
    def __init__(self, path_base, list_id='train', random_crop_size=None, random_ratio=None, random_flip=False, random_rotate=False, gamma=None):
        if list_id == 'train':
            file_list = tuple(open(pjoin(path_base, 'train.txt'), 'r'))
            self.data_list = ['train/' + id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Training Dataset contains %d images' % (self.length))
        elif list_id == 'val':
            file_list = tuple(open(pjoin(path_base, 'val.txt'), 'r'))
            self.data_list = ['val/' + id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Validation Dataset contains %d images' % (self.length))
        elif list_id == 'test':
            file_list = tuple(open(pjoin(path_base, 'test.txt'), 'r'))
            self.data_list = ['test/' +id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Testing Dataset contains %d images' % (self.length))


        self.path_base = pjoin(path_base, 'processed/half')
        self.random_crop_size = random_crop_size
        self.random_flip = random_flip
        self.random_ratio = random_ratio
        self.random_rotate = random_rotate
        self.gamma = gamma
        self.ori_height = 187
        self.ori_width = 620

    def __len__(self):
        return self.length


    def _random_flip(self, x):
        if self.random_flip:
            coin = torch.randint(0, 2, (1,)).item()
            if coin == 0:
                return {'img': TF.hflip(x['img']), 'seg': TF.hflip(x['seg'])}
            else:
                return x
        else:
            return x

    def _random_rotate(self, x):
        if self.random_rotate:
            angle = torch.randint(self.random_rotate[0], self.random_rotate[1], (1,)).item()
            # print(x['seg'].size)
            return {'img': TF.rotate(x['img'], angle, resample=0, fill=(0,0,0)), 
            'seg': TF.rotate(x['seg'], angle, resample=0)}
        else:
            return x
    
    def _random_crop(self, x):       
        if self.random_crop_size:
            ori_height, ori_width = self.ori_height, self.ori_width
            hs = torch.randint(self.random_crop_size[0], self.random_crop_size[1]+1, (1,)).item()
            random_ratio = torch.rand(1).item() * (-self.random_ratio[0] + self.random_ratio[1]) + self.random_ratio[0]
            ws = hs * ori_width / ori_height * random_ratio
            if hs > ori_height or ws > ori_width:
                hp = int(np.abs(-ori_height + hs))
                wp = int(np.abs(-ori_width + ws))
                x_p = torch.randint(0, int(wp+1), (1,)).item()
                y_p = torch.randint(0, int(hp+1), (1,)).item()
                return {'img':TF.resized_crop(TF.pad(x['img'], (wp, hp)), y_p, x_p, hs, ws, [ori_height, ori_width], interpolation=0),
                        'seg':TF.resized_crop(TF.pad(x['seg'], (wp, hp), fill=0), y_p, x_p, hs, ws, [ori_height, ori_width], interpolation=0)}
            else:
                x_p = torch.randint(0, int(ori_width - ws+1), (1,)).item()
                y_p = torch.randint(0, int(ori_height - hs+1), (1,)).item()
                return {'img':TF.resized_crop(x['img'], y_p, x_p, hs, ws, [ori_height, ori_width],interpolation=0),
                 'seg':TF.resized_crop(x['seg'], y_p, x_p, hs, ws, [ori_height, ori_width], interpolation=0)}
        else:
            return x

    def _gamma_transform(self, x):
        if self.gamma:
            gamma = torch.rand(1).item() * (-self.gamma[0] + self.gamma[1]) + self.gamma[0]
            x['img'] = TF.adjust_gamma(x['img'], gamma)
        
        return x


    def __getitem__(self, idx):
        img = np.load(pjoin(self.path_base, self.data_list[idx]) + '_rgb.npy')
        seg = np.load(pjoin(self.path_base, self.data_list[idx]) + '_depth.npy')
        # print(img.shape)
        # print(seg.shape)
        # print(seg.max())
        img = fromarray(img.transpose(1, 2, 0).astype('uint8'))
        
        seg = fromarray((seg * 255. / seg.max()).astype('uint8'))
        # print(seg.size)
   
        x = self._gamma_transform(self._random_rotate(self._random_flip(self._random_crop({'img':img, 'seg':seg}))))
        x['img'] = np.array(x['img'])
        x['seg'] = np.array(x['seg'])

        mask =  ((x['seg'] > 0).astype(np.uint8)) * (1 - (x['img'].mean(2) == 0).astype(np.uint8))
        
        return {'img' : x['img'].transpose(2, 0, 1)/255., 'seg' : x['seg'][None,...]/255., 'mask':mask[None,...], 'img_key':self.data_list[idx]}






if __name__ == '__main__':
    # dataset = CityScapeSwitch('/home/syl/Documents/Project/prob_reg/dataset/cityscape', random_switch=True,random_crop_size=[256, 256], random_ratio=[0.8, 1.25], random_flip=True, random_rotate=[-15, 15], gamma=[0.7,1.5])
    dataset = CityScapeSwitch('/home/syl/Documents/Project/prob_reg/dataset/cityscape', 'val')
    
    # for i in range(10):
    #     fig, ax = plt.subplots(1, 3)
    #     tmp = dataset[100]  
    #     ax[0].imshow(tmp['img'].transpose(1, 2, 0))
    #     ax[1].imshow(dataset.switcher._color_mapping(tmp['seg'][0,...])/255.)
    #     ax[2].imshow(tmp['mask'][0,...])
    #     plt.show()
    # fig, ax = plt.subplots(4, 8)
    # tmp, tmp_prob = dataset.get_gt_modes(0)
    # print(tmp_prob)
    # for i in range(32):
    #     ax[i//8][i%8].imshow(dataset.switcher._color_mapping(tmp[i,...])/255.)
    # plt.show()

    # tmp = dataset[1]
    # seg = tmp['seg']
    # # print(seg.shape)
    # seg = torch.tensor(seg)
    # print(seg.shape)
    # color_seg = dataset.switcher.color_mapping(seg) 
    # print(color_seg.shape)
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(color_seg[0,...].cpu().numpy().transpose(1, 2, 0))
    # plt.show()

    # dataset = MixMNIST('/home/syl/Documents/Project/prob_reg/generated_mu/MNIST', 'test')
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
    #         num_workers=1, pin_memory=True, sampler=None, worker_init_fn= lambda _: torch.manual_seed(0))
    # for i, data in enumerate(data_loader):
        
    #     print(data['img_key'])
    #     fig, ax = plt.subplots(1,2)
    #     ax[0].imshow(data['img'][0,0,...])
    #     ax[1].imshow(data['seg'][0,0,...])
    #     print(data['img'].max())
    #     plt.show()

    # dataset = KittiMonoDepth('/home/syl/Documents/Project/prob_reg/dataset/KITTI/', list_id='train',random_crop_size=[140, 160], random_ratio=[0.8, 1.25], random_flip=True, random_rotate=[-5, 5], gamma=[0.9,1.2])
    # # dataset = CityScapeSwitch('/home/syl/Documents/Project/prob_reg/dataset/cityscape')
    
    # for i in range(10):
    #     fig, ax = plt.subplots(1, 3)
    #     tmp = dataset[33583]  
    #     ax[0].imshow(tmp['img'].transpose(1, 2, 0))
    #     ax[1].imshow(tmp['seg'][0,...])
    #     ax[2].imshow(tmp['mask'][0,...])
    #     plt.show()
