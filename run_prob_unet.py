import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation, AverageMeter
import os
from os.path import join as pjoin
from torchvision.utils import save_image
import argparse
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_training", type=int, default=0, help="training = 1")
    opt = parser.parse_args()


    name = 'prob_unet'
    base_path = '/lustre/project/RonaldLui/Syl/project/prob_reg/'
    save_path = pjoin(base_path, 'models', name)
    image_path = pjoin(base_path, 'images', name)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    print('!!! using {} !!!'.format(device))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    

    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=3, beta=1.0)
    net.to(device)


    def train():
        now = datetime.now()
        date_string = now.strftime("%m_%d_%Y_%H_%M")

        train_dataset = LIDC_IDRI(path_base = '/lustre/project/RonaldLui/Syl/dataset/LIDC/', list_id='train')
        val_dataset = LIDC_IDRI(path_base = '/lustre/project/RonaldLui/Syl/dataset/LIDC/', list_id='val')
        
    
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False,
            num_workers=4, pin_memory=True, sampler=None)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.4)
        epochs = 900
        avg_recon_loss = AverageMeter()
        avg_kll = AverageMeter()
        for epoch in range(epochs):
            avg_recon_loss.reset()
            avg_kll.reset()
            for pg in optimizer.param_groups:
                print('current learning rate {}'.format(pg['lr']))
        

            for step, (patch, mask, _) in enumerate(train_loader): 
                patch = patch.to(device).type(Tensor)
                mask = mask.to(device).type(Tensor)
                # mask = torch.unsqueeze(mask,1)
                # print(patch.shape)
                # print(mask.shape)
                net.forward(patch, mask, training=True)
                recon_loss, kll = net.elbo(mask)
                # reg_loss = l2_regularisation(net.posterior) #+ l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
                loss = recon_loss + kll #+ 1e-5 * reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_recon_loss.update(recon_loss.item(), 1)
                avg_kll.update(kll.item(), 1)
                if (step) % 40 == 0:
                    print('step {} [recon_loss: {:.4f}] [kll: {:.4f}]'.format(step, avg_recon_loss.avg, avg_kll.avg))
            
            scheduler.step()
            
            if (epoch+1) % 10 == 0:
                print('validating ...')
                for tstep, (patch, mask, _) in enumerate(val_loader):
                    # print('validating ...')
                    patch = patch.to(device).type(Tensor)
                    mask = mask.to(device).type(Tensor)
                    # mask = torch.unsqueeze(mask,1)
                    net.forward(patch, mask, training=False)
                    result = net.sample(testing=True)
                    result2 = net.sample(testing=True)
                    result3 = net.sample(testing=True)
                    tmp = torch.cat([patch, result, result2, result3, mask], dim=0)
                    save_image(tmp.data[:, 0, None, :, :], pjoin(image_path, "%d_vmask.png" % tstep), nrow=10, normalize=False)
        
        torch.save(net.state_dict(), pjoin(save_path, date_string + '_epoch_%d.pth'%(epoch+1)))


    def test():
        test_dataset = LIDC_IDRI(path_base = '/lustre/project/RonaldLui/Syl/dataset/LIDC/', list_id='test')
        test_loader = DataLoader(test_dataset, batch_size=10)

        model_saved = '' + 'epoch_%d.pth'%(10)
        model_saved = pjoin(save_path, model_saved)

        print("=> loading checkpoint '{}'".format(model_saved))
        checkpoint = torch.load(model_saved, map_location='cpu')
        # print(checkpoint)
        net.load_state_dict(checkpoint)

        print("=> loaded successfully ")
        del checkpoint
        torch.cuda.empty_cache()

        print('testing ...')
        for tstep, (patch, mask, _) in enumerate(test_loader):
            # print('validating ...')
            patch = patch.to(device).type(Tensor)
            mask = mask.to(device).type(Tensor)
            # mask = torch.unsqueeze(mask,1)
            net.forward(patch, mask, training=False)
            result = net.sample(testing=True)
            tmp = torch.cat([patch, result, mask], dim=0)
            save_image(tmp.data[:, 0, None, :, :], pjoin(image_path, "%d_tmask.png" % tstep), nrow=10, normalize=False)

    if opt.is_training == 1:
        train()
    else:
        test()




















