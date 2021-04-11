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
import scipy.io as sio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeformRandom(Dataset):
    def __init__(self, path_base, list_id='train'):
        if list_id == 'train':
            self.input_data = sio.loadmat(
                pjoin(path_base, 'training', 'input_mu_data.mat')
            )['input_mu_data']
            self.gt_data = sio.loadmat(
                pjoin(path_base, 'training', 'gt_mu_data.mat')
            )['gt_mu_data']
            print('=> Training dataset contains %d images' % self.input_data.shape[0])

        if list_id == 'val':
            self.input_data = sio.loadmat(
                pjoin(path_base, 'testing', 'input_mu_data.mat')
            )['input_mu_data']
            self.gt_data = sio.loadmat(
                pjoin(path_base, 'testing', 'gt_mu_data.mat')
            )['gt_mu_data']
            print('=> Validation dataset contains %d images' % self.input_data.shape[0])

        if list_id == 'test':
            self.input_data = sio.loadmat(
                pjoin(path_base, 'testing', 'input_mu_data.mat')
            )['input_mu_data']
            self.gt_data = sio.loadmat(
                pjoin(path_base, 'testing', 'gt_mu_data.mat')
            )['gt_mu_data']
            print('=> Testing dataset contains %d images' % self.input_data.shape[0])

    def __len__(self):
        return self.input_data.shape[0]

    def num_gt_modes(self):
        return None

    def set_idx2(self, idx2):
        self.idx2 = idx2

    def __getitem__(self, idx):
        if self.idx2 == None:
            img = self.input_data[idx]
            seg = self.gt_data[idx]
            return {'img': img, 'seg' : seg, 'seg_id':0}
        elif isinstance(self.idx2, int):
            img = self.input_data[idx]
            seg = self.gt_data[idx, self.idx2]
            return {'img': img, 'seg' : seg, 'img_key':idx}

    def get_gt_modes(self, img_key):
        seg_modes = np.zeros([20, 50, 50])
        for i in range(20):
            seg_modes[i, ...] = self.gt_data[img_key, i]

        img = self.gt_data[img_key]
        
        return {'gt_modes': seg_modes, 'img' : img[None, ...]}

    
# random.seed(0)
class LIDC_IDRI(Dataset):
    def __init__(self, path_base, list_id='train', random_crop_size=None, random_ratio=None, random_flip=False, random_rotate=False):
        
        if list_id == 'train':
            file_list = tuple(open(pjoin(path_base, 'train.txt'), 'r'))
            self.data_list = [id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Training Dataset contains %d images' % (self.length))
        elif list_id == 'val':
            file_list = tuple(open(pjoin(path_base, 'val.txt'), 'r'))
            self.data_list = [id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Validation Dataset contains %d images' % (self.length))
        elif list_id == 'test':
            file_list = tuple(open(pjoin(path_base, 'test.txt'), 'r'))
            self.data_list = [id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Testing Dataset contains %d images' % (self.length))

        self.path_base = pjoin(path_base, list_id)
        self.idx2 = None

        
        self.random_crop_size = random_crop_size
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_ratio = random_ratio

    def __len__(self):
        return self.length

    def num_gt_modes(self):
        return None

    def set_idx2(self, idx2):
        self.idx2 = idx2

    def _random_crop(self, x):
        
        
        if self.random_crop_size:
            ori_height, ori_width = x['img'].height, x['img'].width 
            hs = torch.randint(int(self.random_crop_size[0]), int(self.random_crop_size[1]+1), (1,)).item()
            random_ratio = torch.rand(1).item() * (-self.random_ratio[0] + self.random_ratio[1]) + self.random_ratio[0]
            ws = hs * ori_width / ori_height * random_ratio
            if hs > ori_height or ws > ori_width:
                hp = int(np.abs(-ori_height + hs))
                wp = int(np.abs(-ori_width + ws))
                x_p = torch.randint(0, int(wp+1), (1,)).item()
                y_p = torch.randint(0, int(hp+1), (1,)).item()
                return {'img':TF.resized_crop(TF.pad(x['img'], (wp, hp)), y_p, x_p, hs, ws, [ori_height, ori_width]),
                        'seg':TF.resized_crop(TF.pad(x['seg'], (wp, hp)), y_p, x_p, hs, ws, [ori_height, ori_width])}
            else:
                x_p = torch.randint(0, int(ori_width - ws + 1), (1,)).item()
                y_p = torch.randint(0, int(ori_height - hs + 1), (1,)).item()
                return {'img':TF.resized_crop(x['img'], y_p, x_p, hs, ws, [ori_height, ori_width]),
                 'seg':TF.resized_crop(x['seg'], y_p, x_p, hs, ws, [ori_height, ori_width])}
        else:
            return x

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
            angle = torch.randint(int(self.random_rotate[0]), int(self.random_rotate[1]+1), (1,)).item()
            return {'img': TF.rotate(x['img'], angle, resample=3, fill=(0,)), 
                    'seg': TF.rotate(x['seg'], angle, resample=3, fill=(0,)) }
        else:
            return x

        
    
    def __getitem__(self, idx):
        if self.idx2 == None:
            seg_id = torch.randint(0, 4, (1,)).item()

            # seg_id = 3
            image_name = pjoin(self.path_base, 'images', self.data_list[idx])
            seg_name = pjoin(self.path_base, 'gt', self.data_list[idx].replace('.png', '_l%d.png'%(seg_id)))

            img = imread(image_name)
            seg = imread(seg_name)
            x = self._random_rotate(self._random_flip( self._random_crop({'img':img, 'seg':seg}) ) )
            img = np.array(x['img'])[None, ...]/ 255.
            seg = np.array(x['seg'])[None, ...]/ 255.

            return {'img': img, 'seg' : seg, 'seg_id':seg_id}
        elif isinstance(self.idx2, int):
            image_name = pjoin(self.path_base, 'images', self.data_list[idx])
            seg_name = pjoin(self.path_base, 'gt', self.data_list[idx].replace('.png', '_l%d.png'%(self.idx2)))

            img = imread(image_name)
            seg = imread(seg_name)
            x = self._random_rotate(self._random_flip(self._random_crop({'img':img, 'seg':seg})))
            img = np.array(x['img'])[None, ...]/ 255.
            seg = np.array(x['seg'])[None, ...]/ 255.

            return {'img':img, 'seg':seg, 'img_key':self.data_list[idx]}

    def get_gt_modes(self, img_key):
        seg_modes = np.zeros([4, 180, 180])
        img_key = img_key[:14] + '/' + img_key[15:]
        for i in range(4):
            seg_modes[i, ...] = np.array(imread(pjoin(self.path_base, 'gt', img_key + '_l%d.png'%(i)))) / 255.

        img = np.array(imread(pjoin(self.path_base, 'images', img_key + '.png'))) / 255.
        
        return {'gt_modes': seg_modes, 'img' : img[None, ...]}


class StochasticLabelSwitches(object):
    """
    Stochastically switches labels in a batch of integer-labeled segmentations.
    """
    def __init__(self):
        self.color_map = {label.trainId:label.color for label in cs_labels_tuple}
        self.color_map[255] = (0.,0.,0.)
        switched_labels2color = {'road_2': (84, 86, 22), 'person_2': (167, 242, 242), 'vegetation_2': (242, 160, 19),
				 		 'car_2': (30, 193, 252), 'sidewalk_2': (46, 247, 180)}


        # trainId2name = {labels.trainId: labels.name for labels in cs_labels_tuple}
        name2trainId = {labels.name: labels.trainId for labels in cs_labels_tuple}

        self._label_switches = OrderedDict([('sidewalk', 8./17.), ('person', 7./17.), ('car', 6./17.), ('vegetation', 5./17.), ('road', 4./17.)])
        
        
        num_classes = 19
        num_classes += len(self._label_switches)
        # switched_Id2name = {19+i:list(self._label_switches.keys())[i] + '_2' for i in range(len(self._label_switches))}
        switched_name2Id = {list(self._label_switches.keys())[i] + '_2':19+i for i in range(len(self._label_switches))}
        switched_cmap = {switched_name2Id[i]:switched_labels2color[i] for i in switched_name2Id.keys()}
        self.color_map = {**self.color_map, **switched_cmap}
        # trainId2name = {**trainId2name, **switched_Id2name}
        self._switched_name2Id = switched_name2Id

        self._name2id = {**name2trainId, **switched_name2Id}

    def __call__(self, data_dict):

        switched_seg = data_dict['seg']

        for c, p in self._label_switches.items():
            init_id = self._name2id[c]
            final_id = self._name2id[c + '_2']
            switch_instance = torch.rand(1).item() <= p


            if switch_instance:
                switched_seg[switched_seg == init_id] = final_id

        data_dict['seg'] = fromarray(switched_seg)
        return data_dict

    def _color_mapping(self, seg):
        color_seg = np.zeros(seg.shape + (3,))
        for c, color in self.color_map.items():
            color_seg[seg == c] = color
        return color_seg

    def color_mapping(self, seg):
        # assume seg of shape (b, h, w)
        color_seg = torch.zeros((seg.shape[0],) + (3,) + seg.shape[1:]).to(device)
        # print(color_seg.shape)
        for c, color in self.color_map.items():
            mask = (seg == c).float()
            color_seg += torch.Tensor(color).to(device).view(1, 3, 1, 1) * mask[:,None,...]
        return color_seg / 255.

    def _switch(self, switched_seg, switch_id):
        i = 0
        prob = 1
        for c, p in self._label_switches.items():
            init_id = self._name2id[c]
            final_id = self._name2id[c + '_2']

            if switch_id[i]:
                switched_seg[switched_seg == init_id] = final_id
                prob = prob * p
            else:
                prob = prob * (1-p)
            i += 1

        return switched_seg, prob



class CityScapeSwitch(Dataset):
    def __init__(self, path_base, list_id='train', random_switch=True, random_crop_size=None, random_ratio=None, random_flip=False, random_rotate=False, gamma=None):
        if list_id == 'train':
            file_list = tuple(open(pjoin(path_base, 'train.txt'), 'r'))
            self.data_list = [id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Training Dataset contains %d images' % (self.length))
        elif list_id == 'val':
            file_list = tuple(open(pjoin(path_base, 'val.txt'), 'r'))
            self.data_list = [id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Validation Dataset contains %d images' % (self.length))
        elif list_id == 'test':
            file_list = tuple(open(pjoin(path_base, 'test.txt'), 'r'))
            self.data_list = [id_.rstrip() for id_ in file_list]
            self.length = len(self.data_list)
            print('=> Testing Dataset contains %d images' % (self.length))


        self.path_base = pjoin(path_base, 'processed/quarter')
        self.random_crop_size = random_crop_size
        self.random_flip = random_flip
        self.random_ratio = random_ratio
        self.random_rotate = random_rotate
        self.switcher = StochasticLabelSwitches()
        self.random_switch = random_switch
        self.gamma = gamma


    def __len__(self):
        return self.length
    

    def num_gt_modes(self):
        return 2**5


    def map_labels_to_trainId(self, arr):
        """Remap ids to corresponding training Ids. Note that the inplace mapping works because id > trainId here!"""
        id2trainId   = {label.id:label.trainId for label in cs_labels_tuple}
        for id, trainId in id2trainId.items():
            arr[arr == id] = trainId
        return arr

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
            angle = torch.randint(int(self.random_rotate[0]), int(self.random_rotate[1]+1), (1,)).item()
            return {'img': TF.rotate(x['img'], angle, resample=0, fill=(0,0,0)), 
            'seg': TF.rotate(x['seg'], angle, resample=0, fill=(255,))}
        else:
            return x
    
    def _random_crop(self, x):       
        if self.random_crop_size:
            ori_height, ori_width = x['img'].height, x['img'].width
            hs = torch.randint(self.random_crop_size[0], self.random_crop_size[1]+1, (1,)).item()
            random_ratio = torch.rand(1).item() * (-self.random_ratio[0] + self.random_ratio[1]) + self.random_ratio[0]
            ws = hs * ori_width / ori_height * random_ratio
            if hs > ori_height or ws > ori_width:
                hp = int(np.abs(-ori_height + hs))
                wp = int(np.abs(-ori_width + ws))
                x_p = torch.randint(0, int(wp+1), (1,)).item()
                y_p = torch.randint(0, int(hp+1), (1,)).item()
                return {'img':TF.resized_crop(TF.pad(x['img'], (wp, hp)), y_p, x_p, hs, ws, [ori_height, ori_width], interpolation=0),
                        'seg':TF.resized_crop(TF.pad(x['seg'], (wp, hp), fill=255), y_p, x_p, hs, ws, [ori_height, ori_width], interpolation=0)}
            else:
                x_p = torch.randint(0, int(ori_width - ws + 1), (1,)).item()
                y_p = torch.randint(0, int(ori_height - hs + 1), (1,)).item()
                return {'img':TF.resized_crop(x['img'], y_p, x_p, hs, ws, [ori_height, ori_width],interpolation=0),
                 'seg':TF.resized_crop(x['seg'], y_p, x_p, hs, ws, [ori_height, ori_width], interpolation=0)}
        else:
            return x

    def _gamma_transform(self, x):
        if self.gamma:
            gamma = self.gamma[0] + torch.rand(1).item() * (-self.gamma[0] + self.gamma[1])
            x['img'] = TF.adjust_gamma(x['img'], gamma)
        
        return x


    def __getitem__(self, idx):
        img = np.load(pjoin(self.path_base, self.data_list[idx]) + '_leftImg8bit.npy')
        seg = np.load(pjoin(self.path_base, self.data_list[idx]) + '_gtFine_labelIds.npy')
        seg = self.map_labels_to_trainId(seg)
        img = fromarray(img.transpose(1, 2, 0).astype('uint8'))
        
        if self.random_switch:
            x = self._gamma_transform(self._random_rotate(self._random_flip(self._random_crop(self.switcher({'img':img, 'seg':seg})))))
        else:
            seg = fromarray(seg)
            x = self._gamma_transform(self._random_rotate(self._random_flip(self._random_crop({'img':img, 'seg':seg}))))
        x['img'] = np.array(x['img'])
        x['seg'] = np.array(x['seg'])

        mask =  (1 - (x['seg'] >= 24).astype(np.uint8)) * (1 - (x['img'].mean(2) == 0).astype(np.uint8))
        
        return {'img' : x['img'].transpose(2, 0, 1)/255., 'seg' : x['seg'][None,...], 'mask':mask[None,...], 'img_key':self.data_list[idx]}

    
    def get_gt_modes(self, img_key):
        gt_modes = np.zeros([32, 256, 512])
        seg = np.load(pjoin(self.path_base, img_key) + '_gtFine_labelIds.npy')
        img = np.load(pjoin(self.path_base, img_key) + '_leftImg8bit.npy')

        seg = self.map_labels_to_trainId(seg)
        probs = np.zeros([32])
        for i in range(32):
            seg_copy = np.copy(seg)
            switch_id = [int(x) for x in format(i, '#07b')[2:]]
            # print(switch_id)
            gt_modes[i], probs[i] = self.switcher._switch(seg_copy, switch_id)
        
        return {'gt_modes': gt_modes, 'img' : img[None, ...], 'prob': probs, 'seg': seg}


class MixMNIST(Dataset):
    def __init__(self, path_base, list_id='train'):
        
        if list_id == 'train': 
            self.data, self.targets = torch.load(pjoin(path_base, 'processed/training.pt'))
            self.length = self.data.shape[0]
            print('=> Dataset contains %d images' % (self.length))
        elif list_id == 'test':
            self.data, self.targets = torch.load(pjoin(path_base, 'processed/test.pt'))
            self.length = self.data.shape[0]
            print('=> Dataset contains %d images' % (self.length))

        self.label_idx_dict = dict()
        for i in range(self.length):
            # print(int(self.targets[i].item()))
            self.label_idx_dict.setdefault(int(self.targets[i].item()), []).append(i)
            
        self.path_base = path_base
        # print(self.label_idx_dict[1])

        self.mode_labels = np.array([[1,2,3,4], [3,4,5,6], [5,6,7,8], [7,8,9,0]])
        self.mode_probs = [0.3, 0.2, 0.2, 0.3]
        self.select_probs = np.array([[0.25, 0.25, 0.25, 0.25], [0.1, 0.4, 0.1, 0.4], [0.3, 0.5, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
        self.cumsum_probs = np.cumsum(self.mode_probs)
        self.cumsum_select_probs = np.cumsum(self.select_probs, axis=1)
        # print(self.cumsum_select_probs)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        # np.random.seed(0)

    def __len__(self):
        return self.length

    def num_gt_modes(self):
        return 16

    def random_sample_from_label(self, label):
        label_list = self.label_idx_dict[label]
        idx = torch.randint(0, len(label_list), (1,)).item()
        return np.array(self.data[label_list[idx]]), idx

    def _mode_sample(self, labels, seg_id, mode_idx):
        img = []
        seg = []
        idxs = []
        assert seg_id in range(4)
        for i in range(4):
            tmp, tmp_idx = self.random_sample_from_label(labels[i])
            img.append(tmp)
            idxs.append(tmp_idx)
            if seg_id == i:
                seg.append(tmp)
            else:
                seg.append(np.zeros_like(tmp))
        img = np.concatenate(img, axis=1)
        seg = np.concatenate(seg, axis=1)
        img = img[None, ...]/255.
        seg = seg[None,...]/255.

        return {'img':img , 'seg':seg, 'img_key': '{}_{}_{}_{}_{}'.format(mode_idx, *idxs), 'mode_id':mode_idx*4+seg_id}


    def __getitem__(self, idx):
        coin = torch.rand(1).item()
        mode_idx = np.argmax(self.cumsum_probs > coin)
        seg_coin = torch.rand(1).item()
        seg_id = np.argmax(self.cumsum_select_probs[mode_idx] > seg_coin)
        # print(mode_idx)
        # print(seg_id)

        return self._mode_sample(self.mode_labels[mode_idx], seg_id, mode_idx)

    
    def get_gt_modes(self, img_key):
        img_key = list(map(int, img_key.split('_')))
        mode_idx = img_key[0]
        idxs = img_key[1:]
        labels = self.mode_labels[mode_idx]
        segs = []
        j = 0
        for label, idx in zip(labels, idxs):
            seg = []
            for i in range(4):
                if i == j:
                    seg.append(np.array(self.data[self.label_idx_dict[label][idx]]))
                else:
                    seg.append(np.zeros_like(np.array(self.data[self.label_idx_dict[label][idx]])))
            j += 1
            seg = np.concatenate(seg, axis=1)
            segs.append(seg[None,...]/255.)
               

        return np.concatenate(segs, axis=0)
            

class MixPosMNIST(Dataset):
    def __init__(self, path_base, list_id='train'):
        
        if list_id == 'train': 
            self.data, self.targets = torch.load(pjoin(path_base, 'processed/training.pt'))
            self.length = self.data.shape[0]
            print('=> Dataset contains %d images' % (self.length))
        elif list_id == 'test':
            self.data, self.targets = torch.load(pjoin(path_base, 'processed/test.pt'))
            self.length = self.data.shape[0]
            print('=> Dataset contains %d images' % (self.length))

        self.label_idx_dict = dict()
        for i in range(self.length):
            # print(int(self.targets[i].item()))
            self.label_idx_dict.setdefault(int(self.targets[i].item()), []).append(i)
            
        self.path_base = path_base
        # print(self.label_idx_dict[1])

        self.mode_labels = np.array([[1,2,3,4], [3,4,5,6], [5,6,7,8], [7,8,9,0]])
        self.mode_probs = [0.25, 0.25, 0.25, 0.25]
        self.select_probs = np.array([[0.25, 0.25, 0.25, 0.25], [0.1, 0.4, 0.1, 0.4], [0.3, 0.5, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
        self.cumsum_probs = np.cumsum(self.mode_probs)
        self.cumsum_select_probs = np.cumsum(self.select_probs, axis=1)
        # print(self.cumsum_select_probs)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        # np.random.seed(0)
        self.input_shape = [28*2, 28*2]

    def __len__(self):
        return self.length

    def num_gt_modes(self):
        return 16

    def random_sample_from_label(self, label):
        label_list = self.label_idx_dict[label]
        idx = torch.randint(0, len(label_list), (1,)).item()
        return np.array(self.data[label_list[idx]]), idx

    def random_sample_top_left_corner(self):
        h_p = torch.randint(0, self.input_shape[0]-28+1, (1,)).item()
        w_p = torch.randint(0, self.input_shape[1]-28+1, (1,)).item()
        return h_p, w_p

    def random_sample_top_left_corner_restricted(self, pos):
        if pos == 0:
            h_p = 0
            w_p = 0
        if pos == 1:
            h_p = 28
            w_p = 0
        if pos == 2:
            h_p = 0
            w_p = 28
        if pos == 3:
            h_p = 28
            w_p = 28

        return h_p, w_p

    def embed_in_pos(self, img, h_p, w_p):
        result = np.zeros(self.input_shape)
        result[h_p:h_p + 28, w_p:w_p+28] = img
        return result

    def _mode_sample(self, labels, seg_id, mode_idx):
        img = np.zeros(self.input_shape)
        seg = np.zeros(self.input_shape)
        idxs = []
        ps = []
        
        assert seg_id in range(4)
        pos = torch.randperm(4)
        for i in range(4):

            tmp, tmp_idx = self.random_sample_from_label(labels[i])
            # tmp_hp, tmp_wp = self.random_sample_top_left_corner()
            tmp_hp, tmp_wp = self.random_sample_top_left_corner_restricted(pos[i])
            tmp = self.embed_in_pos(tmp, tmp_hp, tmp_wp)
            img += tmp
            idxs.append(tmp_idx)
            ps.append((tmp_hp, tmp_wp))
            if seg_id == i:
                seg += tmp
            
        img = img[None, ...]/255.
        seg = seg[None,...]/255.
        img[img > 1] = 1

        img_key = {
            'mode_idx': mode_idx,
            'idxs': idxs,
            'top_left_corners': ps
        }

        return {'img':img , 'seg':seg, 'img_key': img_key}


    def __getitem__(self, idx):
        coin = torch.rand(1).item()
        mode_idx = np.argmax(self.cumsum_probs > coin)
        seg_coin = torch.rand(1).item()
        seg_id = np.argmax(self.cumsum_select_probs[mode_idx] > seg_coin)
        # print(mode_idx)
        # print(seg_id)

        return self._mode_sample(self.mode_labels[mode_idx], seg_id, mode_idx)

    
    def get_gt_modes(self, img_key):
        
        mode_idx = img_key['mode_idx']
        idxs = img_key['idxs']
        ps = img_key['top_left_corners']
        labels = self.mode_labels[mode_idx]
        segs = []
        j = 0
        for label, idx in zip(labels, idxs):
            
            seg = (np.array(self.embed_in_pos(
                self.data[self.label_idx_dict[label][idx]],
                ps[j][0], ps[j][1]
                )
            ))
            
            segs.append(seg[None,...]/255.)
            j += 1
               

        return np.concatenate(segs, axis=0)





# if __name__ == '__main__':
    # dataset = CityScapeSwitch('/home/syl/Documents/Project/prob_reg/dataset/cityscape', random_switch=True,random_crop_size=[256, 256], random_ratio=[0.8, 1.25], random_flip=True, random_rotate=[-15, 15], gamma=[0.7,1.5])
    # dataset = CityScapeSwitch('/home/syl/Documents/Project/prob_reg/dataset/cityscape', 'val')
    
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

    # dataset = MixPosMNIST('/content/gdrive/My Drive/MNIST', 'test')
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
    #         num_workers=1, pin_memory=True, sampler=None, worker_init_fn= lambda _: torch.manual_seed(0))
    # for i, data in enumerate(data_loader):
        
    #     print(data['img_key'])
    #     fig, ax = plt.subplots(1,2)
    #     ax[0].imshow(data['img'][0,0,...])
    #     ax[1].imshow(data['seg'][0,0,...])
    #     print(data['img'].max())
    #     plt.show()