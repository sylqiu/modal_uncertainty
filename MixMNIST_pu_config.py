from loaders import MixMNIST
from probabilistic_unet import ProbabilisticUnet
from os.path import join as pjoin
import torch, torchvision
from torchvision.utils import save_image
import numpy as np

NAME = 'MixMNIST'
use_sigmoid = True
save_base_path = '/lustre/project/RonaldLui/Syl/project/prob_reg/'
input_channels = 1
output_channels = 1
num_filters = [16, 32, 64, 128]
num_feature_channels = 6



net = ProbabilisticUnet(input_channels=input_channels, num_classes=output_channels, num_filters=num_filters, latent_dim=num_feature_channels, no_convs_fcomb=4, beta=10.0)  

train_bs = 256
val_bs = 10
epochs = 250
milestones = [0, 100, 100, 150]
lr_milestones = [1e-4, 5e-5, 1e-5, 5e-6]


resume_training = False
if resume_training:
    resume_check_point = ''

data_path_base = '/lustre/project/RonaldLui/MNIST/'
train_dataset = MixMNIST(path_base = data_path_base, list_id='train')
val_dataset = MixMNIST(path_base = data_path_base, list_id='test')

# for testing #
check_point = 'models/MixMNIST_pu_07_13_2020_16_39/07_13_2020_16_39_epoch_current.pth'
sample_num = 16

save_test_npy = True
if save_test_npy:
    test_bs = 1
else:
    test_bs = 10
test_dataset = MixMNIST(path_base = data_path_base, list_id='test')
test_partial = 100


def save_test(tstep, image_path, batch, patch, ori_seg, recon_seg, sample, prob, sigmoid_layer):
    if 'img_key' in batch.keys():
        img_key = batch['img_key'][0].replace('/', '_').replace('.png', '')
    else:
        img_key = '%d' % (tstep)
    if save_test_npy:
        if output_channels > 1:
            np.save(pjoin(image_path, "{}_{}sample_labelIds.npy".format(img_key, sample_num)), torch.nn.functional.softmax(sample, dim=1).argmax(dim=1).cpu().numpy())
        else:
            sample = sigmoid_layer(sample) > 0.5
            np.save(pjoin(image_path, "{}_{}sample_labelIds.npy".format(img_key, sample_num)), sample.cpu().numpy())
                
            np.save(pjoin(image_path, "{}_{}prob.npy".format(img_key, sample_num)), prob)
    else: 
                
        tmp = torch.cat([patch, ori_seg, sigmoid_layer(recon_seg), sigmoid_layer(sample)], dim=0)
        save_image(tmp.data, pjoin(image_path, "%d.png" % img_key), nrow=test_bs, normalize=False)
