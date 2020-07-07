import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


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
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



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
            logpt = -F.cross_entropy(input, target, reduction='none')
        
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1-pt+1e-2)**self.gamma) * logpt

        return -logpt, loss