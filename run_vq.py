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

    cf = imp.load_source('cf', opt.config + '_config.py')
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
        name =  NAME + '_' + date_string 
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
        avg_code_loss = AverageMeter()
        avg_cls_loss = AverageMeter()
        avg_sigma = AverageMeter()
        avg_mu = AverageMeter()
        if cf.use_focal_loss:
            avg_focal_loss = AverageMeter()

        ###################################
        # calculate initial codebook sigma
        ###################################
        if cf.data_dependant_qstat:
            logging.info('computing initial quatization statistics')
            avg_sigma.reset()
            avg_mu.reset()
            for step, batch in enumerate(train_loader): 

                patch = batch['img'].to(device).type(Tensor)
                seg = batch['seg'].to(device).type(Tensor)
                if 'mask' in batch.keys():
                    mask = batch['mask'].to(device).type(Tensor)
                    # print(mask.shape)
                    # print(seg.shape)
                    seg = seg * mask
                else:
                    mask = None

                avg_sigma.update(net.posterior_forward(patch, seg).pow(2).mean().pow(0.5).item(), 1)
                avg_mu.update(net.posterior_forward(patch, seg).mean().item(), 1)
                if step % 100 == 0:
                    logging.info('mean {}, std {}'.format(avg_mu.avg, avg_sigma.avg))
                if step > 10:
                    break
            logging.info('initial quantization statistics: mean {}, std {}'.format(avg_mu.avg, avg_sigma.avg))
            net._init_emb(mu=avg_mu.avg, sigma=avg_sigma.avg*cf.sigma_scale)
        else:
            net._init_emb()

        ###################################
        # Training
        ###################################
        for epoch in range(cf.epochs):
            avg_seg_loss.reset()
            avg_code_loss.reset()
            avg_cls_loss.reset()
            if cf.use_focal_loss:
                avg_focal_loss.reset()

            if epoch in cf.milestones:
                logging.info('stepping on {}-th learning rate {}'.format(lr_ct, cf.lr_milestones[lr_ct]))
                for pg in optimizer.param_groups:
                    pg['lr'] = cf.lr_milestones[lr_ct]
                lr_ct += 1

                          
            net.train()

            if cf.use_quantization_diff_with_decay:
                diff_decay = cf.init_diff_decay * cf.decay_pow ** epoch
            else:
                diff_decay = 0

            for pg in optimizer.param_groups:
                logging.info('epoch {}: current learning rate {}'.format(epoch+1, pg['lr']))
        
            used_idx = dict()
            for step, batch in enumerate(train_loader): 

                patch = batch['img'].to(device).type(Tensor)
                seg = batch['seg'].to(device).type(Tensor)
                if 'mask' in batch.keys():
                    mask = batch['mask'].to(device).type(Tensor)
                    seg = seg * mask
                else:
                    mask = None

                net.forward(patch, seg, decay=diff_decay)
                ### just for inspection ###
                tmp_idx = net.quantized_posterior_z_ind.cpu().numpy()
                tmp_idx = list(tmp_idx.squeeze())
                for item in tmp_idx:
                    used_idx[item] = used_idx.get(item, 0) + 1
                ### end just for inspection ###
                if cf.use_l1loss:
                    loss_dict = net.l1loss(seg, mask)
                else:
                    loss_dict = net.loss(seg, mask)
                # print(loss_dict)
                
                if epoch >= cf.warm_up_epochs:
                    if cf.use_focal_loss:
                        loss = loss_dict['seg_loss_focal']*cf.focal_weight + loss_dict['code_loss'] * cf.beta + loss_dict['classification_loss']
                    else:
                        loss = loss_dict['seg_loss'] + loss_dict['code_loss'] * cf.beta + loss_dict['classification_loss']
                else:
                    if cf.use_focal_loss:
                        loss = loss_dict['seg_loss_focal']*cf.focal_weight + loss_dict['code_loss'] * cf.beta
                    else:
                        loss = loss_dict['seg_loss'] + loss_dict['code_loss'] * cf.beta


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
                avg_seg_loss.update(loss_dict['seg_loss'].item(), 1)
                avg_code_loss.update(loss_dict['code_loss'].item(), 1)
                avg_cls_loss.update(loss_dict['classification_loss'].item(), 1)
                if cf.use_focal_loss:
                    avg_focal_loss.update(loss_dict['seg_loss_focal'].item(), 1)
                


                if (step) % 40 == 0:
                    if cf.use_focal_loss:
                        logging.info('step {} [seg_loss: {:.4f}/{:.4f}] [focal_loss: {:.4f}/{:.4f}] [code_loss: {:.4f}/{:.4f}] [cls_loss: {:.4f}/{:.4f}] '.format(\
                            step, avg_seg_loss.val, avg_seg_loss.avg, avg_focal_loss.val, avg_focal_loss.avg, avg_code_loss.val, avg_code_loss.avg,  avg_cls_loss.val, avg_cls_loss.avg,
                            )
                        )
                    else:
                        logging.info('step {} [seg_loss: {:.4f}/{:.4f}] [code_loss: {:.4f}/{:.4f}] [cls_loss: {:.4f}/{:.4f}] '.format(\
                            step, avg_seg_loss.val, avg_seg_loss.avg, avg_code_loss.val, avg_code_loss.avg,  avg_cls_loss.val, avg_cls_loss.avg,
                            )
                        )
                
                # break
            code_norm = net.emb.embed.norm(2, dim=0).cpu().numpy()
            logging.info(' quantization usage summary \n ')
            for key in sorted(used_idx.keys()):
                logging.info('{} : {}, '.format(key, used_idx[key]),)
            logging.info(' dictionary size : {}/{}'.format(len(used_idx), net.num_instance))
            logging.info( 'quantization norm \n {}'.format([np.mean(code_norm), np.std(code_norm), np.max(code_norm), np.min(code_norm)]))
            # net.emb._compute_weight(used_idx)
           
            # scheduler.step()

            #### validation ####
            with torch.no_grad():
                if (epoch+1) in cf.milestones + [cf.epochs] or epoch in [0, 2]:
                    net.eval()

                    avg_seg_loss.reset()
                    avg_code_loss.reset()
                    avg_cls_loss.reset()

                    used_idx = dict()
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
                        loss_dict = net.loss(seg, mask)

                        sample1, idx1, _ = net.sample_topk(1)
                        sample2, idx2, _ = net.sample_topk(2)
                        sample3, idx3, _ = net.sample_topk(3)

                        idx = list(idx1.view(-1).cpu().numpy()) + list(idx2.view(-1).cpu().numpy()) + list(idx3.view(-1).cpu().numpy())

                        for item in idx:
                            used_idx[item] = used_idx.get(item, 0) + 1
                        
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
                        avg_code_loss.update(loss_dict['code_loss'].item(), 1)
                        avg_cls_loss.update(loss_dict['classification_loss'].item(), 1)

                        if tstep >= 100 and cf.test_partial:
                            break
                    
        

                    del sample1, sample2, sample3, tmp   

                    logging.info('step {} [seg_loss: {:.4f}] [code_loss: {:.4f}] [cls_loss: {:.4f}'.format(\
                        tstep, avg_seg_loss.avg, avg_code_loss.avg, avg_cls_loss.avg,
                        )
                    )

                    logging.info(' quantization usage summary \n ')
                    for key in sorted(used_idx.keys()):
                        logging.info('{} : {}, '.format(key, used_idx[key]),)

                    logging.info(' dictionary size : {}/{}'.format(len(used_idx), net.num_instance))


            torch.save(net.state_dict(), pjoin(save_path, date_string + '_epoch_current.pth'))
                        

            if epoch+1 in cf.milestones + [cf.epochs]:
                torch.save(net.state_dict(), pjoin(save_path, date_string + '_epoch_%d.pth'%(epoch+1)))

            torch.cuda.empty_cache()


    def test():
        with torch.no_grad():
            torch.manual_seed(0)
            check_point = cf.check_point
            sample_num = cf.sample_num

            name = 'test_'  + NAME + '_' + date_string
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
            
            net._init_emb()
            net.load_state_dict(checkpoint)
            test_dataset = cf.test_dataset
            test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False,
                num_workers=1, pin_memory=True, sampler=None, worker_init_fn= lambda _: torch.manual_seed(0))
            used_idx = dict()
            logging.info('testing ... \n')

            net.eval()
            avg_seg_loss = AverageMeter()
            avg_code_loss = AverageMeter()
            avg_cls_loss = AverageMeter()

            avg_seg_loss.reset()
            avg_code_loss.reset()
            avg_cls_loss.reset()
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
                loss_dict = net.loss(seg, mask)

                logging.info('batch #{} \n seg_loss {} \n  code_loss {} \n cls_loss {} \n '.format(tstep, loss_dict['seg_loss'], loss_dict['code_loss'], loss_dict['classification_loss']))
                
                idx = []
                sample = []
                prob = []
                code_ids = []
                for sample_idx in range(sample_num):
                    if cf.top_k_sample: 
                        sample_tmp, idx_tmp, prob_tmp = net.sample_topk(sample_idx+1)
                    else:
                        sample_tmp, idx_tmp, prob_tmp = net.sample()
                    
                    
                    sample.append(sample_tmp)
                    prob.append(prob_tmp)
                    code_ids.append(idx_tmp.item())

                    del sample_tmp
                    torch.cuda.empty_cache()

                    idx += list(idx_tmp.view(-1).cpu().numpy())
                    # logging.info('sample_{}: [code_id {}] [prob {}] \
                                # '.format(sample_idx, idx_tmp, prob_tmp))

                sample = torch.cat(sample, dim=0)
                
                for item in idx:
                    used_idx[item] = used_idx.get(item, 0) + 1

                if cf.test_partial and tstep > cf.test_partial:
                    break

                
                cf.save_test(tstep, image_path, batch, patch, ori_seg, net.recon_seg, sample, prob, code_ids, sigmoid_layer)
        
                avg_seg_loss.update(loss_dict['seg_loss'].item(), 1)
                avg_code_loss.update(loss_dict['code_loss'].item(), 1)
                avg_cls_loss.update(loss_dict['classification_loss'].item(), 1)

                del sample
                torch.cuda.empty_cache()
                
            
                
            logging.info('step {} [seg_loss: {:.4f}] [code_loss: {:.4f}] [cls_loss: {:.4f}]'.format(\
                tstep, avg_seg_loss.avg, avg_code_loss.avg, avg_cls_loss.avg,
                )
            )

            logging.info(' quantization usage summary \n ')
            for key in sorted(used_idx.keys()):
                logging.info('{} : {}, '.format(key, used_idx[key]),)

            logging.info(' dictionary size : {}/{}'.format(len(used_idx), net.num_instance))







    if opt.is_training == 1:
        train()
    else:
        test()





















