import torch, torchvision
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from loaders import LIDC_IDRI
from vq_models import ResQNet
from utils import l2_regularisation, AverageMeter, sample_from
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from os.path import join as pjoin
from torchvision.utils import save_image
import argparse
from datetime import datetime
import scipy.io as sio
import logging
import imp



if __name__ == '__main__':
    now = datetime.now()
    date_string = now.strftime("%m_%d_%Y_%H_%M")
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_training", type=int, default=0, help="training = 1")
    parser.add_argument("--config", nargs='?', type=str, default='LIDC', help="config name")
    opt = parser.parse_args()

    cf = imp.load_source('cf', opt.config + '_pu_config.py')
    NAME = cf.NAME
    base_path = cf.save_base_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    print('!!! using {} !!!'.format(device))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    

    # net = torch.nn.DataParallel(cf.net)
    net = cf.net
    net.to(device)

    def sigmoid_layer(x):
        if cf.output_channels > 1 and cf.use_sigmoid:
            return cf.train_dataset.switcher.color_mapping(torch.nn.functional.softmax(x, dim=1).argmax(dim=1))
        elif cf.use_sigmoid:
            return  torch.nn.functional.sigmoid(x)
        else:
            return x
            

    def color_mapping(x):
        return cf.train_dataset.switcher.color_mapping(x[:,0,...])

    def scheduler(optimizer,lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer


    def train():
        name =  NAME + '_pu_' + date_string 
        save_path = pjoin(base_path, 'models', name)
        image_path = pjoin(base_path, 'images', name)


        os.makedirs(image_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        logging.basicConfig(filename=image_path + '/LOG.txt',
                            filemode='a',
                            level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info('-------------------------------')
        with open(opt.config + '_config.py', "r") as f:
            logging.info(f.read())
        f.close()
        logging.info('-------------------------------')


        train_dataset = cf.train_dataset
        val_dataset = cf.val_dataset
        

        train_loader = DataLoader(train_dataset,batch_size=cf.train_bs, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)
        val_loader = DataLoader(val_dataset, batch_size=cf.val_bs, shuffle=False,
            num_workers=4, pin_memory=True, sampler=None)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cf.milestones[1:], gamma=cf.lr_decay, last_epoch=-1)
        lr_ct = 0



        avg_seg_loss = AverageMeter()
        avg_kl_loss = AverageMeter()


        ###################################
        # Training
        ###################################
        for epoch in range(cf.epochs):
            avg_seg_loss.reset()
            avg_kl_loss.reset()

            if epoch in cf.milestones:
                logging.info('stepping on {}-th learning rate {}'.format(lr_ct, cf.lr_milestones[lr_ct]))
                for pg in optimizer.param_groups:
                    pg['lr'] = cf.lr_milestones[lr_ct]
                lr_ct += 1

                          
            net.train()


            for pg in optimizer.param_groups:
                logging.info('epoch {}: current learning rate {}'.format(epoch+1, pg['lr']))
        
          
            for step, batch in enumerate(train_loader): 

                patch = batch['img'].to(device).type(Tensor)
                seg = batch['seg'].to(device).type(Tensor)
                if 'mask' in batch.keys():
                    mask = batch['mask'].to(device).type(Tensor)
                    seg = seg * mask
                else:
                    mask = None

                net.forward(patch, seg)


                loss_dict = net.elbo(seg, mask=mask)
                loss = loss_dict['seg_loss'] + loss_dict['kl']
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
                avg_seg_loss.update(loss_dict['seg_loss'].item(), 1)
                avg_kl_loss.update(loss_dict['kl'].item(), 1)


                if (step) % 40 == 0:
                   
                    logging.info('step {} [seg_loss: {:.4f}/{:.4f}] [kl_loss: {:.4f}/{:.4f}] '.format(\
                        step, avg_seg_loss.val, avg_seg_loss.avg, avg_kl_loss.val, avg_kl_loss.avg,
                        )
                    )
                
                # break
            
 
            #### validation ####
            with torch.no_grad():
                if (epoch+1) in cf.milestones + [cf.epochs] or epoch in [0, 2]:
                    net.eval()

                    avg_seg_loss.reset()
                    avg_kl_loss.reset()

                    
                    logging.info('validating ...')
                    for tstep, batch in enumerate(val_loader):
                        # print('validating ...')
                        patch = batch['img'].to(device).type(Tensor)
                        ori_seg = batch['seg'].to(device).type(Tensor)
                        if 'mask' in batch.keys():
                            mask = batch['mask'].to(device).type(Tensor)
                            seg = ori_seg * mask
                        else:
                            mask = None
                            seg = ori_seg

                        net.forward(patch, seg, training=False)
                        loss_dict = net.elbo(seg, mask)

                        sample1 = net.sample()
                        sample2 = net.sample()
                        sample3 = net.sample()

                        
                        if cf.output_channels > 1:
                            tmp = torch.cat([patch, color_mapping(ori_seg), sigmoid_layer(net.recon_seg), sigmoid_layer(sample1), sigmoid_layer(sample2), sigmoid_layer(sample3)], dim=0)
                        else:
                            if patch.shape[1] > 1:
                                tmp = torch.cat([patch, 
                                    ori_seg.expand_as(patch), 
                                    sigmoid_layer(net.recon_seg).expand_as(patch), 
                                    sigmoid_layer(sample1).expand_as(patch), 
                                    sigmoid_layer(sample2).expand_as(patch), 
                                    sigmoid_layer(sample3).expand_as(patch)], dim=0)
                            
                            else:
                                tmp = torch.cat([patch, ori_seg, sigmoid_layer(net.recon_seg), sigmoid_layer(sample1), sigmoid_layer(sample2), sigmoid_layer(sample3)], dim=0)


                        save_image(tmp.data, pjoin(image_path, "%d.png" % tstep), nrow=cf.val_bs, normalize=False)

                        
                        avg_seg_loss.update(loss_dict['seg_loss'].item(), 1)
                        avg_kl_loss.update(loss_dict['kl'].item(), 1)
                       

                        if tstep >= 100 and cf.test_partial:
                            break
                    
        

                    del sample1, sample2, sample3, tmp   

                    logging.info('step {} [seg_loss: {:.4f}] [kl_loss: {:.4f}]'.format(\
                        tstep, avg_seg_loss.avg, avg_kl_loss.avg,      
                        )
                    )



            torch.save(net.state_dict(), pjoin(save_path, date_string + '_epoch_current.pth'))
                        

            if epoch+1 in cf.milestones + [cf.epochs]:
                torch.save(net.state_dict(), pjoin(save_path, date_string + '_epoch_%d.pth'%(epoch+1)))

            torch.cuda.empty_cache()


    def test():

        check_point = cf.check_point
        sample_num = cf.sample_num

        name = 'test_'  + NAME + '_pu_' + date_string
        image_path = pjoin(base_path, 'images', name)
        os.makedirs(image_path, exist_ok=True)

        logging.basicConfig(filename=image_path + '/LOG.txt',
                            filemode='a',
                            level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info('-------------------------------')
        with open(opt.config + '_config.py', "r") as f:
            logging.info(f.read())
        f.close()
        logging.info('-------------------------------')

        # print(net.seg_criterion)

        model_path = pjoin(base_path, check_point)
        checkpoint = torch.load(model_path, map_location='cpu')
        test_bs = cf.test_bs
        logging.info('using check point {} \n'.format(check_point))
        
        net.load_state_dict(checkpoint)
        test_dataset = cf.test_dataset
        test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False,
            num_workers=1, pin_memory=True, sampler=None, worker_init_fn= lambda _: torch.manual_seed(0))
    
        logging.info('testing ... \n')

        net.eval()
        avg_seg_loss = AverageMeter()
        avg_kl_loss = AverageMeter()

        avg_seg_loss.reset()
        avg_kl_loss.reset()

        for tstep, batch in enumerate(test_loader):
            # print('validating ...')
            patch = batch['img'].to(device).type(Tensor)
            ori_seg = batch['seg'].to(device).type(Tensor)
            if 'mask' in batch.keys():
                mask = batch['mask'].to(device).type(Tensor)
                seg = ori_seg * mask
            else:
                mask = None
                seg = ori_seg
            



            net.forward(patch, seg, training=False)
            # print('mask {}'.format(mask))
            loss_dict = net.elbo(seg, mask)

            logging.info('batch #{} \n seg_loss {} \n  kl_loss {} \n '.format(tstep, loss_dict['seg_loss'], loss_dict['kl']))
            
            sample = []
            prob = []

            for sample_idx in range(sample_num):
                sample_tmp = net.sample()  
                prob_tmp = net.z_prior_prob.item()

                sample.append(sample_tmp)
                prob.append(prob_tmp)
               
                logging.info('sample_{}_prob {} \
                            '.format(sample_idx, prob_tmp))
            sample = torch.cat(sample, dim=0)

            if cf.test_partial and tstep > cf.test_partial:
                break

            
            cf.save_test(tstep, image_path, batch, patch, ori_seg, net.recon_seg, sample, prob, sigmoid_layer)
      
            avg_seg_loss.update(loss_dict['seg_loss'].item(), 1)
            avg_kl_loss.update(loss_dict['kl'].item(), 1)
        
            
        logging.info('step {} [seg_loss: {:.4f}] [kl_loss: {:.4f}]'.format(\
            tstep, avg_seg_loss.avg, avg_kl_loss.avg,
            )
        )








    if opt.is_training == 1:
        train()
    else:
        test()





















