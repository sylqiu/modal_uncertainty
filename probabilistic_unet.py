#This code is based on: https://github.com/SimonKohl/probabilistic_unet

from unet_blocks import *
from unet import *
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation, sobel_gradient
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from nearest_embed import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, x):
        out = self.layers(x)
        return out

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, inputs, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = inputs
            self.show_seg = segm
            inputs = torch.cat((inputs, segm), dim=1)
            self.show_concat = inputs
            self.sum_input = torch.sum(inputs)

        encoding = self.encoder(inputs)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=[2,3], keepdim=True)
        # encoding = torch.mean(encoding, dim=3, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            self.layers.apply(init_weights)
            self.last_layer.apply(init_weights)


    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = z.view(-1, self.latent_dim, 1, 1) * torch.ones([z.shape[0], self.latent_dim, feature_map.shape[2], feature_map.shape[3]]).to('cuda')           

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            out = self.layers(feature_map)
            return self.last_layer(out)


class ZClassifier(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_feature_channels, num_output_channels, no_convs_fcomb):
        super(ZClassifier, self).__init__()
        self.num_output_channels = num_output_channels #output channels
        self.num_input_channels = num_feature_channels
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.no_convs_fcomb = no_convs_fcomb 

        layers = []

        #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
        layers.append(nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb-2):
            layers.append(nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.last_layer = nn.Conv2d(self.num_input_channels, self.num_output_channels, kernel_size=1)

        self.layers.apply(init_weights)
        self.last_layer.apply(init_weights)


    def forward(self, feature_map):
        out = self.layers(feature_map)
        return self.last_layer(out)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, apply_last_layer=False, padding=True).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, use_tile=True).to(device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch,False)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            #z_prior = self.prior_latent_space.base_dist.loc 
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.sigmoid(self.fcomb.forward(self.unet_features,z_prior))


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = nn.BCEWithLogitsLoss(reduction=None)
        z_posterior = self.posterior_latent_space.rsample()
        
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)
        
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss) / segm.shape[0]
        # self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return self.reconstruction_loss, self.beta * self.kl


class QNet(nn.Module):

    def __init__(self, input_channels=1, num_filters=[32,64,128,192], no_convs_fcomb=4, beta=10.0):
        super(QNet, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.latent_dim = num_filters[-1]
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.beta = beta
        self.num_instance = 4096

        self.unet_encoder_full = EncoderSimp(self.input_channels+1, self.num_filters, padding=True).to(device)
        self.unet_decoder_recon = Decoder(self.input_channels+1, self.num_filters, apply_last_layer=True, padding=True)
        # self.unet_decoder_recon = Decoder(self.input_channels, self.num_filters, apply_last_layer=True, padding=True)
        # self.unet_decoder_seg = Decoder(self.input_channels, self.num_filters, apply_last_layer=True, padding=True)

        self.emb = Quantize(self.latent_dim, self.num_instance)

        self.unet_encoder_part = EncoderSimp(self.input_channels, self.num_filters, padding=True).to(device)
        self.classifier = ZClassifier(self.num_filters[-1], self.num_instance, self.no_convs_fcomb).to(device)

        self.recon_criterion = nn.L1Loss(reduction='mean')
        self.seg_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.classification_criterion = nn.CrossEntropyLoss(reduction='mean')

        self.opt_group1 =nn.ModuleList([ 
                        self.unet_encoder_full,
                        self.unet_decoder_recon,
                        self.emb,
        ])

        self.opt_group2 = nn.ModuleList([
                        self.unet_encoder_part,
                        self.classifier,
                        
        ])

        self.sobel_gradient = sobel_gradient()

    def forward_train1(self, patch, segm):

        self.posterior_z, size_list = self.unet_encoder_full.forward(torch.cat([patch, segm], dim=1))
        # self.unet_decoder_recon.size_list = size_list
        # self.unet_decoder_seg.size_list = size_list
        self.unet_decoder_recon.size_list = size_list[:-1]
        self.quantized_posterior_z, self.l2_diff, self.quantized_posterior_z_ind = self.emb(self.posterior_z)
        # print('!!')
        # print(self.quantized_posterior_z.shape)
        
        # self.recon_img = self.unet_decoder_recon.forward(self.quantized_posterior_z) 
        # self.recon_seg = self.unet_decoder_seg.forward(self.quantized_posterior_z) 
        self.recon_input = self.unet_decoder_recon.forward(self.quantized_posterior_z)
        self.recon_img = self.recon_input[:,0,None,...]
        self.recon_seg = self.recon_input[:,1,None,...] 


    def forward_train2(self, patch):        
        self.prior_z, _ = self.unet_encoder_part.forward(patch)
        self.prior_z_ind = self.classifier(self.prior_z)


    def loss_train1(self, patch, segm):
        recon_grad = self.sobel_gradient.compute(self.recon_img-patch)
        recon_loss = self.recon_criterion(input=self.recon_img, target=patch) + self.recon_criterion(input=recon_grad, target=torch.zeros_like(recon_grad, requires_grad=False))
        seg_loss = self.seg_criterion(input=self.recon_seg, target=segm)
       
        # reconstruction_loss = (recon_loss * 2 + seg_loss) / 2
       
        return recon_loss, seg_loss, self.beta * self.l2_diff
        
    def loss_train2(self):
        classification_loss = self.classification_criterion(self.prior_z_ind, self.quantized_posterior_z_ind)

        return classification_loss


class HQNet(nn.Module):

    def __init__(self, input_channels=1, no_convs_fcomb=4, beta=10.0):
        super(HQNet, self).__init__()
        self.input_channels = input_channels
        self.num_filters_1 = [32, 64]
        self.num_filters_1_D = [32, 64, 64+64]
        self.num_filters_2 = [128, 192]
        self.num_filters_2_D = [64, 128, 192]
        self.latent_dim_1 = self.num_filters_1[-1]
        self.latent_dim_2 = self.num_filters_2[-1]

        self.no_convs_fcomb = no_convs_fcomb
        self.beta = beta
        self.num_instance_1 = 128
        self.num_instance_2 = 256

        self.unet_encoder_full_1 = EncoderSimp(self.input_channels+1, self.num_filters_1, padding=True, no_first_pool=False).to(device)
        self.unet_encoder_full_2 = EncoderSimp(self.latent_dim_1, self.num_filters_2, padding=True, no_first_pool=False).to(device)

        self.unet_decoder_recon_2 = Decoder(None, self.num_filters_2_D, apply_last_layer=False, padding=True)
        self.unet_decoder_recon_1 = Decoder(self.input_channels+1, self.num_filters_1_D, apply_last_layer=True, padding=True)

        self.emb_1 = Quantize(self.latent_dim_1, self.num_instance_1)
        self.emb_2 = Quantize(self.latent_dim_2, self.num_instance_2)

        self.unet_encoder_part_1 = EncoderSimp(self.input_channels, self.num_filters_1, padding=True,  no_first_pool=False).to(device)
        self.unet_encoder_part_2 = EncoderSimp(self.latent_dim_1, self.num_filters_2, padding=True, no_first_pool=False).to(device)

        self.classifier_1 = ZClassifier(self.latent_dim_1, self.num_instance_1, self.no_convs_fcomb).to(device)
        self.classifier_2 = ZClassifier(self.latent_dim_2, self.num_instance_2, self.no_convs_fcomb).to(device)

        self.recon_criterion = nn.L1Loss(reduction='mean')
        self.seg_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.classification_criterion = nn.CrossEntropyLoss(reduction='mean')

        self.sobel_gradient = sobel_gradient()

        self.opt_group1 =nn.ModuleList([
                        self.unet_encoder_full_1, 
                        self.unet_encoder_full_2,
                        self.unet_decoder_recon_1,
                        self.unet_decoder_recon_2,
                        self.emb_1,
                        self.emb_2,
        ])

        self.opt_group2 = nn.ModuleList([
                        self.unet_encoder_part_1,
                        self.unet_encoder_part_2,
                        self.classifier_1,
                        self.classifier_2
                        
        ])

    def forward_train1(self, patch, segm):


        self.posterior_z_1, size_list_1 = self.unet_encoder_full_1.forward(torch.cat([patch, segm], dim=1))
        self.unet_decoder_recon_1.size_list = size_list_1[:-1]
        # print(self.unet_decoder_recon_1.size_list )

        self.posterior_z_2, size_list_2 = self.unet_encoder_full_2.forward(self.posterior_z_1)
        self.unet_decoder_recon_2.size_list = size_list_2[:-1]
        # print(self.unet_decoder_recon_2.size_list)

        self.quantized_posterior_z_1, self.l2_diff_1, self.quantized_posterior_z_ind_1 = self.emb_1(self.posterior_z_1)
        self.quantized_posterior_z_2, self.l2_diff_2, self.quantized_posterior_z_ind_2 = self.emb_2(self.posterior_z_2)

        # print(self.quantized_posterior_z_1.shape)
        # print(self.quantized_posterior_z_2.shape)
        
        intermediate_feature = self.unet_decoder_recon_2.forward(self.quantized_posterior_z_2)
        self.recon_input = self.unet_decoder_recon_1.forward(torch.cat([self.quantized_posterior_z_1, intermediate_feature], dim=1))

        self.recon_img = self.recon_input[:,0,None,...]
        self.recon_seg = self.recon_input[:,1,None,...] 


    def forward_train2(self, patch):
            
        self.prior_z_1, _ = self.unet_encoder_part_1.forward(patch)
        self.prior_z_ind_1 = self.classifier_1(self.prior_z_1)
        self.prior_z_2, _ = self.unet_encoder_part_2.forward(self.prior_z_1)
        self.prior_z_ind_2 = self.classifier_2(self.prior_z_2)


    def loss_train1(self, patch, segm):
        recon_grad = self.sobel_gradient.compute(self.recon_img-patch)
        recon_loss = self.recon_criterion(input=self.recon_img, target=patch) + self.recon_criterion(input=recon_grad, target=torch.zeros_like(recon_grad, requires_grad=False))
        seg_loss = self.seg_criterion(input=self.recon_seg, target=segm)
       
        # reconstruction_loss = (recon_loss * 2 + seg_loss) / 2
        
        

        return recon_loss, seg_loss, \
                self.beta * (self.l2_diff_1 + self.l2_diff_2)
                

    def loss_train2(self):
        classification_loss_1 = self.classification_criterion(self.prior_z_ind_1, self.quantized_posterior_z_ind_1.detach())
        classification_loss_2 = self.classification_criterion(self.prior_z_ind_2, self.quantized_posterior_z_ind_2.detach())
        return classification_loss_1 + classification_loss_2