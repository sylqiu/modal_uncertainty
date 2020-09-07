import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import collections
from PIL import Image


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm+1e-5)
        return out


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = dict()
    self.avg = dict()
    self.sum = dict()
    self.count = 0

  def update(self, val):
    self.val = val
    self.count += 1
    for key in val:
      self.sum[key] = self.sum.get(key, 0) + val[key].item()
      self.avg[key] = self.sum[key] / self.count

def log_loss_dict(step, mean_loss_dict):
  log_str = 'step {} '.format(step)
  for key in mean_loss_dict:
    log_str += '[{}] {} '.format(key, mean_loss_dict[key])
  logging.info(log_str)


class CreateFrequencySummarizer():
    def __init__(self, table_size):
        
        self.table_size = table_size
        self.table = np.zeros(self.table_size) # this is fine if we only use small table size 

    def log_in_table(self, item_attribute_idx):
        self.table[tuple(item_attribute_idx)] += 1

    def _normalize(self, table, axis=None):
        if axis==None:
            return table / np.sum(table)
        else:
            return table / np.sum(table, axis=axis)

    def normalize(self):
        self.table = self._normalize(self.table)

    def retain_axis(self, axis):
        '''
        sum over the complmentary axes
        '''
        one_hot = [True if tmp in axis else False for tmp in range(self.table_size)]
        sum_axis = np.array(range(len(self.table_size)))[one_hot]
        return np.sum(self.table, tuple(sum_axis))
        
    def visualize_two_axis(self, axis1, axis2, normalize=True, cmap='viridis'):
        new_table = self._normalize(self.retain_axis([axis1, axis2]))
        cm = plt.get_cmap(cmap)
        new_table_img = Image.fromarray(np.array(cm(new_table)[...,:3]))
        new_table_img = new_table_img.resize((self.table_size[axis1]*2, self.table_size[axis2]*2), Image.NEAREST)
        return new_table, new_table_img

    


class CreateResultSaver(object):
    '''
    save scalars, and tensors into numpy 
    save images into png
    '''
    def __init__(self, name, base_dir, token, cmap='viridis'):
        
        self.name = name
        self.parent_path = os.path.join(base_dir, name, token)
        self.image_path = os.path.join(base_dir, name, token, 'images')
        self.quant_path = os.path.join(base_dir, name, token, 'quant')
        os.makedirs(self.parent_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.quant_path, exist_ok=True)
        self.scalar_dict = collections.defaultdict(list)
        self.tensor_dict = collections.defaultdict(list)
        self.cmap = plt.get_cmap(cmap)
        

    def write_image(self, step, image_dict):
        img_save = []
        dict_len = len(image_dict)
        for key, value in image_dict.items():
            img_array = []
            value = value.numpy()
            bdim = value.shape[0]
            for i in range(bdim):
                img = value[i]
                if img.shape[-1] == 1:
                    img = img[...,0]
                    img = self.cmap(img)[...,:3] # only get the RGB
                elif img.shape[-1] == 3:
                    img = img
                else:
                    raise NotImplementedError
                
                # pad the image only when there are mutiple images to save
                if dict_len > 1 or bdim > 1:
                    img = np.pad(img, ((3, 3), (3, 3), (0, 0)))
                # print(img.shape)
                img_array.append(img)
            img_array = np.concatenate(img_array, axis=1)
            img_save.append(img_array)
        
        img_save = np.concatenate(img_save, axis=0)
        plt.imsave(os.path.join(self.image_path, '{}.png'.format(step)), img_save)

    def append(self, step, scalar_dict=None, tensor_dict=None):
        if scalar_dict is not None:
            for key, value in scalar_dict.items():
                self.scalar_dict[key].append(value.numpy())
        if tensor_dict is not None:
            for key, value in tensor_dict.items():
                self.tensor_dict[key].append(value.numpy())
    
    def save_dict_to_numpy(self):
        if bool(self.scalar_dict):
            for key, value in self.scalar_dict:
                np.save(os.path.join(self.quant_path, key+'_scalar.npy'), np.stack(value))
        if bool(self.tensor_dict):
            for key, value in self.tensor_dict:
                np.save(os.path.join(self.quant_path, key+'_tensor.npy'), np.stack(value))


def sobel(window_size):
	assert(window_size%2!=0)
	ind=window_size // 2
	matx=[]
	maty=[]
	for j in range(-ind,ind+1):
		row=[]
		for i in range(-ind,ind+1):
			if (i*i+j*j)==0:
				gx_ij=0
			else:
				gx_ij=i/float(i*i+j*j)
			row.append(gx_ij)
		matx.append(row)
	for j in range(-ind,ind+1):
		row=[]
		for i in range(-ind,ind+1):
			if (i*i+j*j)==0:
				gy_ij=0
			else:
				gy_ij=j/float(i*i+j*j)
			row.append(gy_ij)
		maty.append(row)

	# matx=[[-3, 0,+3],
	# 	  [-10, 0 ,+10],
	# 	  [-3, 0,+3]]
	# maty=[[-3, -10,-3],
	# 	  [0, 0 ,0],
	# 	  [3, 10,3]]
	if window_size==3:
		mult=2
	elif window_size==5:
		mult=20
	elif window_size==7:
		mult=780

	matx=np.array(matx)*mult				
	maty=np.array(maty)*mult

	print('!! sobel')
	print(matx)
	print(maty)

	return torch.Tensor(matx), torch.Tensor(maty)

def create_window(window_size, channel):
	windowx,windowy = sobel(window_size)
	windowx,windowy= windowx.unsqueeze(0).unsqueeze(0), windowy.unsqueeze(0).unsqueeze(0)
	windowx = torch.Tensor(windowx.expand(channel,1,window_size,window_size))
	windowy = torch.Tensor(windowy.expand(channel,1,window_size,window_size))
	# print windowx
	#print windowy

	return windowx,windowy

class sobel_gradient(object):
    def __init__(self,channel=1):
        self.channel=channel
        self.windowx, self.windowy = create_window(3, self.channel)
        self.windowx = self.windowx.cuda()
        self.windowy = self.windowy.cuda()
    
    def compute(self, img, padding=0):
        if self.channel > 1 :		# do convolutions on each channel separately and then concatenate
            gradx=torch.ones(img.shape)
            grady=torch.ones(img.shape)
            for i in range(self.channel):
                gradx[:,i,:,:]=F.conv2d(img[:,i,:,:].unsqueeze(0), self.windowx, padding=padding,groups=1).squeeze(0)   #fix the padding according to the kernel size
                grady[:,i,:,:]=F.conv2d(img[:,i,:,:].unsqueeze(0), self.windowy, padding=padding,groups=1).squeeze(0)

        else:
            gradx = F.conv2d(img, self.windowx, padding=padding,groups=1)
            grady = F.conv2d(img, self.windowy, padding=padding,groups=1)

        return torch.cat([gradx, grady], dim=1)
    

def sample_from(probas):
    N, level_count, h, w = probas.size()
    val = torch.rand(N, 1, h ,w)
    if probas.is_cuda:
        val = val.cuda()
    cutoffs = torch.cumsum(probas, dim=1)
    _, idx = torch.max(cutoffs > val, dim=1)
    value = torch.gather(probas, 1, idx.unsqueeze(1))
    # print(value.shape)
    # print(idx.shape)
    # out = idx.float() / (level_count - 1)
    return value.squeeze(1), idx


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau



class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0.5):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma


    def forward(self, input, target):


        # compute the negative likelyhood
        if input.shape[1] == 1:
            logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')        
        else:
            logpt = -F.cross_entropy(input, target[:,0,...].long(), reduction='none')
        
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1-pt+1e-2)**self.gamma) * logpt

        return -logpt, loss