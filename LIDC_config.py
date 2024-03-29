from loaders import LIDC_IDRI
from vq_models import ResQNet
from os.path import join as pjoin
import torch, torchvision
from torchvision.utils import save_image
import numpy as np

NAME = 'LIDC'
use_sigmoid = True
save_base_path = '/lustre/project/RonaldLui/Syl/project/prob_reg/'
input_channels = 1
output_channels = 1
num_filters = [32, 64, 128, 192]
num_classifier_filters = [256, 512]
num_instance = 512
num_feature_channels = 128
posterior_layer = -1
use_focal_loss = False
if use_focal_loss:
    focal_weight = 1
use_l1loss = False

data_dependant_qstat = True
if data_dependant_qstat:
    sigma_scale = 1
use_quantization_diff_with_decay = False
if use_quantization_diff_with_decay:
    init_diff_decay = 0.8
    decay_pow = 0.97

net = ResQNet(input_channels=input_channels, output_channels=output_channels, \
                num_filters=num_filters, num_classifier_filters=num_classifier_filters, num_instance=num_instance, num_feature_channels=num_feature_channels, 
                posterior_layer=posterior_layer, bn=False)  

train_bs = 32
val_bs = 10
epochs = 900
milestones = [0, 300, 600, 800]
lr_milestones = [1e-4, 5e-5, 1e-5, 5e-6]
warm_up_epochs = 150
beta = 0.25
lr_decay = 0.3

resume_training = False
if resume_training:
    resume_check_point = ''

data_path_base = '/lustre/project/RonaldLui/Syl/dataset/LIDC/'
train_dataset = LIDC_IDRI(path_base = data_path_base, list_id='train', random_crop_size=[160,200], random_ratio=[0.8, 1/0.8], random_flip=True, random_rotate=[-20, 20])
val_dataset = LIDC_IDRI(path_base = data_path_base, list_id='val')

# for testing #
check_point = 'models/LIDC_07_16_2020_21_28/07_16_2020_21_28_epoch_current.pth'
sample_num = 16
top_k_sample = True

save_test_npy = True
if save_test_npy:
    test_bs = 1
else:
    test_bs = 5
test_dataset = LIDC_IDRI(path_base = data_path_base, list_id='test')
test_dataset.idx2 = 0
test_partial = 200

def save_test(tstep, image_path, batch, patch, ori_seg, recon_seg, sample, prob, code_ids, sigmoid_layer):
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
            np.save(pjoin(image_path, "{}_{}code_id.npy".format(img_key, sample_num)), code_ids)
    else: 
                
        tmp = torch.cat([patch, ori_seg, sigmoid_layer(recon_seg), sigmoid_layer(sample)], dim=0)
        save_image(tmp.data, pjoin(image_path, "%d.png" % img_key), nrow=test_bs, normalize=False)