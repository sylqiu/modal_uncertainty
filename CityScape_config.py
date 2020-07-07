from loaders import CityScapeSwitch
from vq_models import ResQNet

NAME = 'CityScape'
save_base_path = '/lustre/project/RonaldLui/Syl/project/prob_reg/'
input_channels = 3
output_channels = 24
num_filters = [32, 64, 128, 192, 192, 192]
num_classifier_filters = [256, 512]
num_instance = 512
num_feature_channels = 64
posterior_layer = 1
use_focal_loss = False
if use_focal_loss:
    focal_weight = 1

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

train_bs = 10
val_bs = 4
epochs = 500
milestones = [0, 100, 300, 450]
lr_milestones = [1e-4, 5e-5, 1e-5, 5e-6]
warm_up_epochs = 50
beta = 0.25
lr_decay = 0.3

resume_training = False
if resume_training:
    resume_check_point = ''

data_path_base = '/lustre/project/RonaldLui/CityScape'
train_dataset = CityScapeSwitch(path_base = data_path_base, list_id='train', random_crop_size=[250,270], random_ratio=[0.8, 1.25], random_flip=True, random_rotate=[-15, 15], gamma=[0.7, 1.5])
val_dataset = CityScapeSwitch(path_base = data_path_base, list_id='val')

# for testing #
check_point = 'models/CityScape_07_04_2020_17_02/07_04_2020_17_02_epoch_current.pth'
sample_num = 16
top_k_sample = True

save_test_npy = True
if save_test_npy:
    test_bs = 1
else:
    test_bs = 5
test_dataset = val_dataset
