from unet_blocks import *
from unet import *
from utils import l2_regularisation, sobel_gradient, Normalize, FocalLoss2d
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from nearest_embed import *
from copy import deepcopy
from torch.autograd import Variable
import numpy as np
# import logging
np.set_printoptions(precision=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logging.getLogger().addHandler(logging.StreamHandler())



class Classifier(nn.Module):
    def __init__(self, num_filters=[192, 256, 512], num_feature_channels=128, num_instance=4096, T=0.07):
        super(Classifier, self).__init__()
        self.num_filters = num_filters
        self.num_instance = num_instance
        self.num_layers = len(self.num_filters)
        self.num_feature_channels = num_feature_channels
        self.l2norm = Normalize()
        self.T = T

        layers = []

        for i in range(self.num_layers1):
            layers.append(DownConvBlock(self.num_filters[i], self.num_filters[i+1]))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(self.num_filters[-1], self.num_feature_channels, kernel_size=1, padding=0))
        self.layers = nn.Sequential(*layers)
        
        self.dict = Variable(torch.randn(self.num_instance, self.num_feature_channels),requires_grad=True).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def _set_dict(self, x):
        self.dict = Variable(x.transpose(0, 1).data, requires_grad=True).to(device)

    def forward(self, feature_map):
        # self.feature = self.l2norm(self.layers(feature_map)).div_(self.T)
        self.feature = self.layers(feature_map)

        self.logit = F.linear(self.feature.view(-1, self.num_feature_channels), self.dict.view(self.num_instance, self.num_feature_channels))


    def loss(self, posterior_z_ind):
        
        # print(self.logit.shape)
        # print(posterior_z_ind.shape)
        loss = self.criterion(self.logit, posterior_z_ind)
        # self.dict.grad.data.zero_()
        return loss

    def _backward_update_dict(self, lr):
        # print(torch.sum(torch.abs(self.dict.grad.data)))
        # print(lr)
        self.dict.data -= 0.1*lr * self.dict.grad.data
        

class NaiveClassifier(nn.Module):
    def __init__(self, num_filters=[192, 256, 512], num_feature_channels=512, num_instance=4096, T=0.07):
        super(NaiveClassifier, self).__init__()
        self.num_filters = num_filters
        self.num_instance = num_instance
        self.num_layers = len(self.num_filters)
        self.num_feature_channels = num_feature_channels
        self.l2norm = Normalize()
        self.T = T

        layers = []

        for i in range(self.num_layers-1):
            layers.append(DownConvBlock(self.num_filters[i], self.num_filters[i+1]))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(self.num_filters[-1], self.num_feature_channels, kernel_size=1, padding=0))

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(self.num_feature_channels, self.num_instance)
        # self.dict = Variable(torch.randn(self.num_instance, self.num_feature_channels),requires_grad=True).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, feature_map):
        # self.feature = self.l2norm(self.layers(feature_map)).div_(self.T)
        self.feature = self.layers(feature_map)
        self.logit = self.linear(self.feature.view(-1, self.num_feature_channels))


    def loss(self, posterior_z_ind):
        
        # print(self.logit.shape)
        # print(posterior_z_ind.shape)
        loss = self.criterion(self.logit, posterior_z_ind)
        # self.dict.grad.data.zero_()
        return loss
        

class ResClassifier(nn.Module):
    def __init__(self, num_filters=[192, 256, 512], num_feature_channels=128, num_instance=4096, bn=False):
        super(ResClassifier, self).__init__()
        self.num_filters = num_filters
        self.num_instance = num_instance
        self.num_layers = len(self.num_filters)
        self.num_feature_channels = num_feature_channels
        self.bn = bn

        layers = []


        for i in range(self.num_layers-1):
            layers.append(DownConvBlockRes(self.num_filters[i], self.num_filters[i+1], bn=self.bn))

        layers.append(nn.AdaptiveAvgPool2d(1))
        # layers.append(nn.Conv2d(self.num_filters[-1], self.num_feature_channels, kernel_size=1, padding=0, bias=False))
        self.layers = nn.Sequential(*layers)
        
        # self.dict = Variable(torch.randn(self.num_instance, self.num_feature_channels),requires_grad=True).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.linear = nn.Linear(self.num_filters[-1], self.num_instance)


    def _set_dict(self, x):
        self.dict = Variable(x.transpose(0, 1).data).to(device)

    def forward(self, feature_map):
        # self.feature = self.l2norm(self.layers(feature_map)).div_(self.T)
        self.feature = self.layers(feature_map)

        # self.logit = F.linear(self.feature.view(-1, self.num_feature_channels), self.dict.view(self.num_instance, self.num_feature_channels)) #\
            # - self.feature.norm(2,dim=1).view(-1, 1) \
            # - self.dict.norm(2, dim=1).view(1, -1)
        
        # print(self.logit)
        self.logit = self.linear(self.feature.view(-1, self.num_filters[-1]))
        
        # self.logit = self.logit / self.num_feature_channels


    def loss(self, posterior_z_ind):
        
        # print(self.logit.shape)
        # print(posterior_z_ind.shape)
        loss = self.criterion(self.logit, posterior_z_ind)
        # self.dict.grad.data.zero_()
        return loss



class PosteriorNet(nn.Module):
    def __init__(self, input_channels, num_filters=[32, 64, 128, 192, 256, 512], num_feature_channels=128):
        super(PosteriorNet, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_feature_channels = num_feature_channels
        self.contracting_path = nn.ModuleList()
        

        self.contracting_path.append(nn.Conv2d(input_channels, num_filters[0], kernel_size=3, padding=1))

        for i in range(len(self.num_filters)-1):
            input_dim = self.num_filters[i]
            output_dim = self.num_filters[i+1]
            self.contracting_path.append(DownConvBlock(input_dim, output_dim, pool=True))
        
        self.contracting_path.append(nn.AdaptiveAvgPool2d(1))
        self.contracting_path.append(nn.Conv2d(self.num_filters[-1], self.num_feature_channels, kernel_size=1, padding=0))


    def forward(self, x):
        for i, down in enumerate(self.contracting_path):
            x = down(x)
        # x = self.l2norm(x).div_(self.T)
        return x

class ResPosteriorNet(nn.Module):
    def __init__(self, input_channels, num_filters=[32, 64, 128, 192, 256, 512], num_feature_channels=512, bn=False):
        super(ResPosteriorNet, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_feature_channels = num_feature_channels
        self.contracting_path = nn.ModuleList()
        self.bn = bn
        

        # self.contracting_path.append(nn.Conv2d(input_channels, num_filters[0], kernel_size=3, padding=1))
        
        for i in range(len(self.num_filters)):
            if i == 0:
                pool = False
            else:
                pool = True
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]
            
            self.contracting_path.append(DownConvBlockRes(input_dim, output_dim, pool=pool, bn=self.bn))
        
        self.contracting_path.append(nn.AdaptiveAvgPool2d(1))
        self.contracting_path.append(nn.Conv2d(self.num_filters[-1], self.num_feature_channels, kernel_size=1, padding=0, bias=False))
        # self.linear = nn.Linear(self.num_filters[-1]*4, self.num_features_channels, bias=False)


    def forward(self, x):
        for i, down in enumerate(self.contracting_path):
            x = down(x)
        # x = self.linear(x.view(-1, self.num_filters[-1]*4))
        # x = x.view(-1, self.num_feature_channels, 1, 1)
        return x

# class SpatialDecoder(nn.Module):
#     def __init__(self, num_filters=[128, 128, 128, 128], num_feature_channels=512, bn=False):
#         super(SpatialDecoder, self).__init__()
#         self.num_filters = num_filters
#         self.num_feature_channels = num_feature_channels
#         self.bn = bn

#         self.expanding_path = nn.ModuleList()
#         for i in range(len(self.num_filters)-1):
#             input_dim = self.num_filters[i]
#             output_dim = self.num_filters[i+1]
#             self.contracting_path.append(UpConvBlockRes(input_dim, output_dim, pool=True, bn=self.bn))


class QNet(nn.Module):

    def __init__(self, input_channels=1, output_channels=1, num_filters=[32,64,128,192], num_feature_channels=192):
        super(QNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        
        self.latent_dim = num_feature_channels
        self.num_filters_D = deepcopy(self.num_filters)
        self.num_filters_D[-1] += self.latent_dim

        self.num_instance = 1024*4
        print('using {} quantisations'.format(self.num_instance))
        

        self.unet_encoder = UnetEncoder(self.input_channels, self.num_filters).to(device)
        self.unet_decoder = UnetDecoder(self.output_channels, self.num_filters_D, apply_last_layer=True).to(device)

        self.posterior_encoder = PosteriorNet(self.input_channels+1, num_feature_channels=self.latent_dim).to(device)
        self.emb = Quantize(self.latent_dim, self.num_instance).to(device)
        self.classifier = Classifier(num_feature_channels=self.latent_dim, num_instance=self.num_instance).to(device)
        # self.classifier = ZClassifier(num_instance=self.num_instance, T=self.T).to(device)

        # self.recon_criterion = nn.L1Loss(reduction='mean')
        self.seg_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.sm = nn.Softmax(dim=1)


    def forward(self, patch, segm):

        self.prior_z, self.blocks = self.unet_encoder.forward(torch.cat([patch], dim=1))
        self.posterior_z = self.posterior_encoder.forward(torch.cat([patch, segm], dim=1))

        self.quantized_posterior_z, self.l2_diff, self.quantized_posterior_z_ind = self.emb(self.posterior_z)
        self.classifier._set_dict(self.emb.embed.detach())
        self.classifier.forward(self.prior_z)

        self.quantized_posterior_z = F.interpolate(self.quantized_posterior_z, [self.prior_z.shape[2], self.prior_z.shape[3]])
        intermediate_feature = torch.cat([self.prior_z, self.quantized_posterior_z], dim=1)
        # intermediate_feature = torch.cat([self.quantized_posterior_z], dim=1)
        self.recon_seg = self.unet_decoder.forward(intermediate_feature, self.blocks)

    def sample(self):
        probas = self.sm(self.classifier.logit)
        N, level_count = probas.size()
        val = torch.rand(N, 1)
        if probas.is_cuda:
            val = val.cuda()
        cutoffs = torch.cumsum(probas, dim=1)
        _, idx = torch.max(cutoffs > val, dim=1)
        print('sample probability {}'.format(probas.gather(1, idx.view(N, 1)).squeeze().detach().cpu().numpy()))
        # out = idx.float() / (level_count - 1)
        samples = F.embedding(idx, self.emb.embed.transpose(0,1))
        samples = samples.view(-1, self.latent_dim, 1, 1)
        samples = F.interpolate(samples, [self.prior_z.shape[2], self.prior_z.shape[3]])
        intermediate_feature = torch.cat([self.prior_z, samples], dim=1)
        # intermediate_feature = torch.cat([samples], dim=1)
        return self.unet_decoder.forward(intermediate_feature, self.blocks)

    def sample_topk(self, top_k):
        probas = self.sm(self.classifier.logit)
        N, level_count = probas.size()
        _, idx = torch.topk(probas, top_k, dim=1)
        idx = idx[:, top_k-1]
        print('top {} probability {}'.format(top_k, probas.gather(1, idx.view(N, 1)).squeeze().detach().cpu().numpy()))
        # out = idx.float() / (level_count - 1)
        samples = F.embedding(idx, self.emb.embed.transpose(0,1))
        samples = samples.view(-1, self.latent_dim, 1, 1)
        samples = F.interpolate(samples, [self.prior_z.shape[2], self.prior_z.shape[3]])
        intermediate_feature = torch.cat([self.prior_z, samples], dim=1)
        # intermediate_feature = torch.cat([samples], dim=1)
        return self.unet_decoder.forward(intermediate_feature, self.blocks)

     
    def loss(self, segm):
        seg_loss = self.seg_criterion(self.recon_seg, segm)
        code_loss = self.l2_diff
        classification_loss = self.classifier.loss(self.quantized_posterior_z_ind.squeeze())

        return seg_loss, code_loss, classification_loss


class ResQNet(nn.Module):

    def __init__(self, input_channels=1, output_channels=1, num_filters=[32,64,128,192], num_classifier_filters=[192], num_instance=128, num_feature_channels=320, posterior_layer=-1, bn=False):
        super(ResQNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.bn = bn
        self.classifier_filters = [num_filters[-1]] + num_classifier_filters
        
        self.latent_dim = num_feature_channels
        self.posterior_layer = posterior_layer
        self.num_instance = num_instance
        print('using {} quantisations'.format(self.num_instance))
        
        self.unet_encoder = ResUnetEncoder(self.input_channels, self.num_filters, bn=self.bn)
        self.unet_decoder = ResUnetDecoder(self.output_channels, self.num_filters, posterior_layer=self.posterior_layer, posterior_dim=self.latent_dim, apply_last_layer=True, bn=self.bn)

        self.posterior_encoder = ResPosteriorNet(self.input_channels+self.output_channels, num_filters=self.num_filters+self.classifier_filters[1:], num_feature_channels=self.latent_dim, bn=self.bn)
        
        self.classifier = ResClassifier(num_feature_channels=self.latent_dim, num_filters=self.classifier_filters, num_instance=self.num_instance, bn=self.bn)

        # if self.output_channels == 1:
        #     self.seg_criterion = nn.BCEWithLogitsLoss(reduction='none')
        #     print('use binary cross_entropy')
        # else:
        #     self.seg_criterion = nn.CrossEntropyLoss(reduction='none')
        #     print('use cross_entropy')
        self.seg_criterion_focal = FocalLoss2d(gamma=0.5)
        self.sm = nn.Softmax(dim=1)


    def _init_emb(self, mu=0, sigma=1):
        self.emb = QuantizeEMA(self.latent_dim, self.num_instance, sigma=sigma).to(device)


    def forward(self, patch, segm, decay=0, training=True):

        self.prior_z, self.blocks = self.unet_encoder.forward(torch.cat([patch], dim=1))
        if self.output_channels > 1:
            segm = F.one_hot(segm[:,0,...].long(), self.output_channels).permute(0,3,1,2) - 0.5
            # print(segm.shape)
        self.posterior_z = self.posterior_encoder.forward(torch.cat([patch, segm], dim=1))

        self.quantized_posterior_z, self.diff, self.quantized_posterior_z_ind = self.emb(self.posterior_z, training=training)
        
        # self.classifier._set_dict(self.emb.embed.detach())
        self.classifier.forward(self.prior_z)

        # self.quantized_posterior_z = F.interpolate(self.quantized_posterior_z, [self.prior_z.shape[2], self.prior_z.shape[3]])
        # intermediate_feature = torch.cat([self.prior_z, self.quantized_posterior_z], dim=1)
        # intermediate_feature = torch.cat([self.quantized_posterior_z], dim=1)
        self.recon_seg = self.unet_decoder.forward(self.prior_z, self.quantized_posterior_z + decay*self.diff, self.blocks)
      

    def posterior_forward(self, patch, segm): 
        if self.output_channels > 1:
            segm = F.one_hot(segm[:,0,...].long(), self.output_channels).permute(0,3,1,2) - 0.5  
        return self.posterior_encoder.forward(torch.cat([patch, segm], dim=1))


    def sample(self, effective_idx=None, verbose=False):
        used_logit = self.classifier.logit
        if effective_idx:
            used_logit = used_logit[:, effective_idx, ...]
        probas = self.sm(used_logit)
        N, level_count = probas.size()
        val = torch.rand(N, 1)
        if probas.is_cuda:
            val = val.cuda()
        cutoffs = torch.cumsum(probas, dim=1)
        _, idx = torch.max(cutoffs > val, dim=1)
        probas_idx = probas.gather(1, idx.view(N, 1)).squeeze().detach().cpu().numpy()
        if verbose:
            print('sample probability {}'.format(probas.gather(1, idx.view(N, 1)).squeeze().detach().cpu().numpy()))
        # out = idx.float() / (level_count - 1)
        emb_idx = idx
        if effective_idx:
            for j in range(N):
                emb_idx[j] = effective_idx[idx[j]]
        samples = F.embedding(emb_idx, self.emb.embed.transpose(0,1))
        samples = samples.view(-1, self.latent_dim, 1, 1)
        # samples = F.interpolate(samples, [self.prior_z.shape[2], self.prior_z.shape[3]])
        # intermediate_feature = torch.cat([self.prior_z, samples], dim=1)
        # intermediate_feature = torch.cat([samples], dim=1)
        return self.unet_decoder.forward(self.prior_z, samples, self.blocks), emb_idx, probas_idx

    def sample_topk(self, top_k, verbose=False):
        probas = self.sm(self.classifier.logit)
        N, level_count = probas.size()
        _, idx = torch.topk(probas, top_k, dim=1)
        idx = idx[:, top_k-1]
        probas_idx = probas.gather(1, idx.view(N, 1)).squeeze().detach().cpu().numpy()
        if verbose:
            print('top {} probability {}'.format(top_k, probas_idx))
            print('top {} idx {}'.format(top_k, idx))
        # out = idx.float() / (level_count - 1)
        samples = F.embedding(idx, self.emb.embed.transpose(0,1))
        samples = samples.view(-1, self.latent_dim, 1, 1)
        # samples = F.interpolate(samples, [self.prior_z.shape[2], self.prior_z.shape[3]])
        # intermediate_feature = torch.cat([self.prior_z, samples], dim=1)
        # intermediate_feature = torch.cat([samples], dim=1)
        return self.unet_decoder.forward(self.prior_z, samples, self.blocks), idx, probas_idx
     
    def loss(self, segm, mask=None):
        if mask is not None:
            # seg_loss = self.seg_criterion(self.recon_seg, segm[:,0,...].long())
            # print(seg_loss.shape)
            # print(mask.shape)
            # seg_loss = seg_loss * mask[:,0,...]
            seg_loss, seg_loss_focal = self.seg_criterion_focal(self.recon_seg, segm)
            # print(seg_loss_focal.shape) 
            seg_loss = seg_loss * mask[:,0,...]
            seg_loss_focal = seg_loss_focal * mask[:,0,...]
            mask_sum = mask.sum()
            seg_loss = seg_loss.sum() / (mask_sum + 1e-5)
            seg_loss_focal = seg_loss_focal.sum() / (mask_sum + 1e-5)
           
        else:
            # seg_loss = self.seg_criterion(self.recon_seg, segm)
            seg_loss, seg_loss_focal = self.seg_criterion_focal(self.recon_seg, segm)
            seg_loss = seg_loss.mean()
            seg_loss_focal = seg_loss_focal.mean()

        
        code_loss = self.diff.pow(2).mean()
        classification_loss = self.classifier.loss(self.quantized_posterior_z_ind.view(-1))

        return {'seg_loss':seg_loss, 'seg_loss_focal':seg_loss_focal, 'code_loss':code_loss, 'classification_loss':classification_loss}

    def l1loss(self, segm, mask=None):
        if mask is not None:
            # seg_loss = self.seg_criterion(self.recon_seg, segm[:,0,...].long())
            # print(seg_loss.shape)
            # print(mask.shape)
            # seg_loss = seg_loss * mask[:,0,...]
            seg_loss = F.smooth_l1_loss(self.recon_seg, segm, reduction='none')
            # print(seg_loss_focal.shape) 
            seg_loss = seg_loss * mask[:,0,...]
            mask_sum = mask.sum()
            seg_loss = seg_loss.sum() / (mask_sum + 1e-5)
           
        else:
            # seg_loss = self.seg_criterion(self.recon_seg, segm)
            seg_loss = F.smooth_l1_loss(self.recon_seg, segm, reduction='none')
            seg_loss = seg_loss.mean()

        
        code_loss = self.diff.pow(2).mean()
        classification_loss = self.classifier.loss(self.quantized_posterior_z_ind.view(-1))

        return {'seg_loss':seg_loss, 'code_loss':code_loss, 'classification_loss':classification_loss}






class MultiScaleResQNet(nn.Module):

    def __init__(self, input_channels=1, output_channels=1, num_filters=[32,64,128,192], num_feature_channels=192, bn=False):
        super(MultiScaleResQNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.bn = bn
        
        self.latent_dim = num_feature_channels
        self.num_filters_D = deepcopy(self.num_filters)
        self.num_filters_D[-1] += self.latent_dim

        self.num_instance = 1024*4
        print('using {} quantisations'.format(self.num_instance))
        
        self.unet_encoder = ResUnetEncoder(self.input_channels, self.num_filters, bn=self.bn).to(device)
        self.unet_decoder = MultiScaleResUnetDecoder(self.output_channels, self.num_filters_D, apply_last_layer=True, bn=self.bn).to(device)

        self.posterior_encoder = ResPosteriorNet(self.input_channels+1, num_feature_channels=self.latent_dim, bn=self.bn).to(device)
        self.emb = Quantize(self.latent_dim, self.num_instance).to(device)
        self.classifier = ResClassifier(num_feature_channels=self.latent_dim, num_instance=self.num_instance, bn=self.bn).to(device)
            
        self.seg_criterion = MultiScaleBCELoss()
        self.sm = nn.Softmax(dim=1)


    def forward(self, patch, segm):

        self.prior_z, self.blocks = self.unet_encoder.forward(torch.cat([patch], dim=1))
        self.posterior_z = self.posterior_encoder.forward(torch.cat([patch, segm], dim=1))

        self.quantized_posterior_z, self.l2_diff, self.quantized_posterior_z_ind = self.emb(self.posterior_z)
        self.classifier._set_dict(self.emb.embed)
        self.classifier.forward(self.prior_z)

        self.quantized_posterior_z = F.interpolate(self.quantized_posterior_z, [self.prior_z.shape[2], self.prior_z.shape[3]])
        intermediate_feature = torch.cat([self.prior_z, self.quantized_posterior_z], dim=1)
        # intermediate_feature = torch.cat([self.quantized_posterior_z], dim=1)
        self.recon_seg, self.multiscale_predictions = self.unet_decoder.forward(intermediate_feature, self.blocks)

    def sample(self):
        probas = self.sm(self.classifier.logit)
        N, level_count = probas.size()
        val = torch.rand(N, 1)
        if probas.is_cuda:
            val = val.cuda()
        cutoffs = torch.cumsum(probas, dim=1)
        _, idx = torch.max(cutoffs > val, dim=1)
        print('sample probability {}'.format(probas.gather(1, idx.view(N, 1)).squeeze().detach().cpu().numpy()))
        # out = idx.float() / (level_count - 1)
        samples = F.embedding(idx, self.emb.embed.transpose(0,1))
        samples = samples.view(-1, self.latent_dim, 1, 1)
        samples = F.interpolate(samples, [self.prior_z.shape[2], self.prior_z.shape[3]])
        intermediate_feature = torch.cat([self.prior_z, samples], dim=1)
        # intermediate_feature = torch.cat([samples], dim=1)
        return self.unet_decoder.forward(intermediate_feature, self.blocks)[0]

    def sample_topk(self, top_k):
        probas = self.sm(self.classifier.logit)
        N, level_count = probas.size()
        _, idx = torch.topk(probas, top_k, dim=1)
        idx = idx[:, top_k-1]
        print('top {} probability {}'.format(top_k, probas.gather(1, idx.view(N, 1)).squeeze().detach().cpu().numpy()))
        # out = idx.float() / (level_count - 1)
        samples = F.embedding(idx, self.emb.embed.transpose(0,1))
        samples = samples.view(-1, self.latent_dim, 1, 1)
        samples = F.interpolate(samples, [self.prior_z.shape[2], self.prior_z.shape[3]])
        intermediate_feature = torch.cat([self.prior_z, samples], dim=1)
        # intermediate_feature = torch.cat([samples], dim=1)

        return self.unet_decoder.forward(intermediate_feature, self.blocks)[0]
     
    def loss(self, segm):
        seg_loss = self.seg_criterion(self.multiscale_predictions, segm)
        code_loss = self.l2_diff
        classification_loss = self.classifier.loss(self.quantized_posterior_z_ind.squeeze())

        return seg_loss, code_loss, classification_loss





if __name__ == '__main__':
    patch = torch.rand(2, 1, 180, 180).to(device)
    segm = torch.rand(2, 1, 180, 180).to(device)

    net = QNet()

    net.forward(patch, segm)
    seg_loss, code_loss, classification_loss = net.loss(segm)
    print(net.sample().shape)
    