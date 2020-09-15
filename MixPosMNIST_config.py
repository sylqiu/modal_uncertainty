from loaders import MixPosMNIST
from vq_models import ResQNet
from os.path import join as pjoin
import torch, torchvision
from torchvision.utils import save_image
import numpy as np

NAME = 'MixPosMNIST'
use_sigmoid = True
save_base_path = './results'
input_channels = 1
output_channels = 1
num_filters = [32, 64, 128, 128, 128]
num_classifier_filters = [256, 256]
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

train_bs = 256
val_bs = 10
epochs = 250
milestones = [0, 100, 150, 200]
lr_milestones = [1e-4, 3e-5, 1e-5, 3e-6]
warm_up_epochs = 30
beta = 0.25
lr_decay = 0.3

resume_training = False
if resume_training:
    resume_check_point = ''

data_path_base = './dataset/MNIST'
train_dataset = MixPosMNIST(path_base = data_path_base, list_id='train')
val_dataset = MixPosMNIST(path_base = data_path_base, list_id='test')

# for testing #
check_point = ''
sample_num = 11
top_k_sample = True

save_test_npy = True
if save_test_npy:
    test_bs = 1
else:
    test_bs = 10
test_dataset = MixPosMNIST(path_base = data_path_base, list_id='test')
test_partial = 100

use_result_saver = True
use_frequency_summarizer = False
frequency_table_size=[num_instance, train_dataset.num_gt_modes()]
def get_item_attribute_idx(batch, primary_code_id):
    return (primary_code_id, batch['mode_id'])

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
