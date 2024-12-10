import logging
import os
from functools import partial
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import yaml
from torch.utils.data import DataLoader

from utils.adversarial_trainer import GANFactory
from utils.metric_counter import MetricCounter
from utils.tools import seq_backwarp, tensor2im
from loss.losses import ReconLoss, EPELoss, TVLoss, PhotoMetricLoss
from loss.patchnce import PatchNCELoss, PatchNCELoss2
from models.models import get_model
from utils.schedulers import LinearDecay, WarmRestart
from fire import Fire
from models.reconflownet_cl2 import ReconFlowNet_SNNencoder_dilation
from models.flownet import PWCNet, backwarp
import flowiz as fz
import time
import lpips

class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader, exp=None, load_model=None):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.lambda_epe = config['lambda_epe']
        self.lambda_recon = config['lambda_recon']
        self.lambda_tv = config['lambda_tv']
        self.lambda_cl = config['lambda_cl']
        self.lambda_pm = config['lambda_pm']
        self.scale = 3
        self.load_model = load_model
        exp_name = exp if exp is not None else time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.exp_path = os.path.join(config['experiment_desc'], time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))+'test-normalflow-wbigobject') #default 或 pm1 -normalflow-wobigobject
        self.metric_counter = MetricCounter(self.exp_path)
        self.warmup_epochs = config['warmup_num']

    def train(self):
        self._init_params()

        for epoch in range(self.config['num_epochs']):
            #self._check_dataset(epoch)
            #state_dict = torch.load('/home/zhulin/reconflow/code_0205/vidar_flow_recon/experiments/11_01_11_20_26ReconFlowNet_SNNencoder_dilation-bs2-4411-final/best_model.h5')
            #self.netG.load_state_dict(state_dict['model'])
            self._run_epoch(epoch)
            #self._validate_lpips(epoch)
            
            self._validate(epoch)
            self.scheduler_G.step()

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.netG.state_dict()
                }, os.path.join(self.exp_path, 'best_model.h5'))
            
            if epoch%5 == 0:
                torch.save({
                    'model': self.netG.state_dict()
                }, os.path.join(self.exp_path, '{:04d}.h5'.format(int(epoch))))

            # torch.save({
            #         'model': self.netF.state_dict()
            #     }, os.path.join(self.exp_path, 'pwcnet'))
            print(self.metric_counter.loss_message())
            logging.info("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.exp_path.split('/')[-1], epoch, self.metric_counter.loss_message()))
        print()
    def clip_gradient(self, optimizer, grad_clip):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def _check_dataset(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = self.config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            spike, frame_gt, flow_f_gt, flow_b_gt, occ_f, occ_b = data
            self.model.get_images(spike, frame_gt, flow_f_gt, flow_b_gt, occ_f, occ_b, '/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/check_dataset/')
            
            
    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = self.config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            # inputs, targets = self.model.get_input(data)
            self.optimizer_G.zero_grad()
            spike, frame_gt, flow_f_gt, flow_b_gt, occ_f, occ_b = data
            spike = spike[:, 2:-2,:,:]
            frame_f_gt, frame_b_gt = frame_gt[:,0], frame_gt[:,2]
            
            # flops, params = get_model_complexity_info(self.netG, (43, 448, 256), as_strings=True, print_per_layer_stat=False)  # 不⽤写batch_size⼤⼩，默认batch_size=1
            # print('Flops:  ' + flops)
            # print('Params: ' + params)
            
            
            flows_f_pd, flows_b_pd, recons_f_pd, recons_b_pd = self.netG(spike)
            #feats_enc = self.netG.encoder(spike, 'spike', mode='cl')
            #feats_img = self.netG.encoder(torch.cat([recons_f_pd[-1],recons_b_pd[-1]], 0), 'image', mode='cl')

            loss_recon = self.criterionRecon(frame_f_gt, recons_f_pd) + self.criterionRecon(frame_b_gt, recons_b_pd) if self.lambda_recon>0 else torch.zeros(1).cuda()
            loss_epe = self.criterionEPE(flow_f_gt, flows_f_pd) + self.criterionEPE(flow_b_gt, flows_b_pd) if self.lambda_epe>0 else torch.zeros(1).cuda()
            loss_tv = self.criterionTV(flows_f_pd[-1]) + self.criterionTV(flows_b_pd[-1]) if self.lambda_tv>0 else torch.zeros(1).cuda()
            loss_pm = self.criterionPM(recons_f_pd[-1], recons_b_pd[-1], flows_f_pd[-1], flows_b_pd[-1], occ_f, occ_b) if self.lambda_pm>0 else torch.zeros(1).cuda()
            #loss_cl = self.criterionCL(feats_img, feats_enc) if self.lambda_cl>0 else torch.zeros(1).cuda()
            
            loss_G = self.lambda_recon*loss_recon + \
                        self.lambda_epe*loss_epe + \
                        self.lambda_tv*loss_tv + \
                        self.lambda_pm*loss_pm #+ \
                        #self.lambda_cl*loss_cl

            loss_G.backward()
            self.optimizer_G.step()

            self.metric_counter.add_losses(loss_G.item(), loss_recon.item(), loss_epe.item(), loss_pm.item())#, loss_cl.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(recons_f_pd[-1], 
                                                                                  recons_b_pd[-1], 
                                                                                  flows_f_pd[-1],
                                                                                  flows_b_pd[-1],
                                                                                  frame_f_gt,
                                                                                  frame_b_gt,
                                                                                  flow_f_gt,
                                                                                  flow_b_gt)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            functional.reset_net(self.netG)
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = self.config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        with torch.no_grad():
            for data in tq:
                spike, frame_gt, flow_f_gt, flow_b_gt, occ_f, occ_b = data
                frame_f_gt, frame_b_gt = frame_gt[:,0], frame_gt[:,2]
                spike = spike[:, 32:-32,:,:]
                spike = spike.repeat(2,1,1,1)
                flows_f_pd, flows_b_pd, recons_f_pd, recons_b_pd = self.netG(spike)
                #feats_enc = self.netG.encoder(spike, 'spike', mode='cl')
                #feats_img = self.netG.encoder(torch.cat([recons_f_pd[-1],recons_b_pd[-1]], 0), 'image', mode='cl')
                
                for chl in range(2):
                    flows_f_pd[chl] = flows_f_pd[chl][0].unsqueeze(0)
                    flows_b_pd[chl] = flows_b_pd[chl][0].unsqueeze(0)
                    recons_f_pd[chl] = recons_f_pd[chl][0].unsqueeze(0)
                    recons_b_pd[chl] = recons_b_pd[chl][0].unsqueeze(0)
                    
                loss_recon = self.criterionRecon(frame_f_gt, recons_f_pd) + self.criterionRecon(frame_b_gt, recons_b_pd) if self.lambda_recon>0 else torch.zeros(1).cuda()
                loss_epe = self.criterionEPE(flow_f_gt, flows_f_pd) + self.criterionEPE(flow_b_gt, flows_b_pd) if self.lambda_epe>0 else torch.zeros(1).cuda()
                loss_tv = self.criterionTV(flows_f_pd[-1]) + self.criterionTV(flows_b_pd[-1]) if self.lambda_tv>0 else torch.zeros(1).cuda()
                loss_pm = self.criterionPM(recons_f_pd[-1], recons_b_pd[-1], flows_f_pd[-1], flows_b_pd[-1], occ_f, occ_b) if self.lambda_pm>0 else torch.zeros(1).cuda()
                #loss_cl = self.criterionCL(feats_img, feats_enc) if self.lambda_cl>0 else torch.zeros(1).cuda()
                
                loss_G = self.lambda_recon*loss_recon + \
                         self.lambda_epe*loss_epe + \
                         self.lambda_tv*loss_tv + \
                         self.lambda_pm*loss_pm #+ \
                         #self.lambda_cl*loss_cl

                self.metric_counter.add_losses(loss_G.item(), loss_recon.item(), loss_epe.item(), loss_pm.item())#, loss_cl.item())
                curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(recons_f_pd[-1], 
                                                                                    recons_b_pd[-1], 
                                                                                    flows_f_pd[-1],
                                                                                    flows_b_pd[-1],
                                                                                    frame_f_gt,
                                                                                    frame_b_gt,
                                                                                    flow_f_gt,
                                                                                    flow_b_gt)
                

                self.metric_counter.add_metrics(curr_psnr, curr_ssim)
                if not i:
                    self.metric_counter.add_image(img_for_vis, tag='val')
                i += 1
                functional.reset_net(self.netG)
                if i > epoch_size:
                    break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _validate_lpips(self, epoch):
        self.metric_counter.clear()
        epoch_size = self.config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        lpips_all = 0
        psnr_all = 0
        ssim_all = 0
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        with torch.no_grad():
            for data in tq:
                spike, frame_gt, flow_f_gt, flow_b_gt, occ_f, occ_b = data
                frame_f_gt, frame_b_gt = frame_gt[:,0], frame_gt[:,2]
                #spike = spike[:, 32:-32,:,:]
                spike = spike.repeat(2,1,1,1)
                flows_f_pd, flows_b_pd, recons_f_pd, recons_b_pd = self.netG(spike)
                #feats_enc = self.netG.encoder(spike, 'spike', mode='cl')
                #feats_img = self.netG.encoder(torch.cat([recons_f_pd[-1],recons_b_pd[-1]], 0), 'image', mode='cl')
                
                for chl in range(2):
                    flows_f_pd[chl] = flows_f_pd[chl][0].unsqueeze(0)
                    flows_b_pd[chl] = flows_b_pd[chl][0].unsqueeze(0)
                    recons_f_pd[chl] = recons_f_pd[chl][0].unsqueeze(0)
                    recons_b_pd[chl] = recons_b_pd[chl][0].unsqueeze(0)
                    
                loss_recon = self.criterionRecon(frame_f_gt, recons_f_pd) + self.criterionRecon(frame_b_gt, recons_b_pd) if self.lambda_recon>0 else torch.zeros(1).cuda()
                loss_epe = self.criterionEPE(flow_f_gt, flows_f_pd) + self.criterionEPE(flow_b_gt, flows_b_pd) if self.lambda_epe>0 else torch.zeros(1).cuda()
                loss_tv = self.criterionTV(flows_f_pd[-1]) + self.criterionTV(flows_b_pd[-1]) if self.lambda_tv>0 else torch.zeros(1).cuda()
                loss_pm = self.criterionPM(recons_f_pd[-1], recons_b_pd[-1], flows_f_pd[-1], flows_b_pd[-1], occ_f, occ_b) if self.lambda_pm>0 else torch.zeros(1).cuda()
                #loss_cl = self.criterionCL(feats_img, feats_enc) if self.lambda_cl>0 else torch.zeros(1).cuda()
                
                loss_G = self.lambda_recon*loss_recon + \
                         self.lambda_epe*loss_epe + \
                         self.lambda_tv*loss_tv + \
                         self.lambda_pm*loss_pm #+ \
                         #self.lambda_cl*loss_cl
                         
                 # best forward scores
                #loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

                img0 = frame_f_gt #torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
                img1 = recons_f_pd[-1][0]
                d1 = loss_fn_alex(img0.repeat((1,3,1,1)), img1.repeat((1,3,1,1)))
                
                img0 = frame_b_gt #torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
                img1 = recons_b_pd[-1][0]
                d2 = loss_fn_alex(img0.repeat((1,3,1,1)), img1.repeat((1,3,1,1)))                
                
                lpips_loss = (d1+d2)/2
                
                lpips_all += lpips_loss
                
                
                
                
                
                
                self.metric_counter.add_losses(lpips_loss.item(), loss_recon.item(), loss_epe.item(), loss_pm.item())#, loss_cl.item())
                curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics2(recons_f_pd[-1], 
                                                                                    recons_b_pd[-1], 
                                                                                    flows_f_pd[-1],
                                                                                    flows_b_pd[-1],
                                                                                    frame_f_gt,
                                                                                    frame_b_gt,
                                                                                    flow_f_gt,
                                                                                    flow_b_gt)
                psnr_all += curr_psnr
                ssim_all += curr_ssim

                self.metric_counter.add_metrics(curr_psnr, curr_ssim)
                if not i:
                    self.metric_counter.add_image(img_for_vis, tag='val')
                i += 1
                functional.reset_net(self.netG)
                if i > epoch_size:
                    break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)
        
    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        loss_D = self.adv_lambda_d*self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionRecon = ReconLoss()
        self.criterionEPE = EPELoss()
        self.criterionTV = TVLoss()
        self.criterionCL = PatchNCELoss2()
        self.criterionPM = PhotoMetricLoss()

        # self.netG = SpikeNet(32,11,7).cuda()
        self.netG = ReconFlowNet_SNNencoder_dilation().cuda()
        if self.load_model is not None:
            state_dict = torch.load(self.load_model)
            # state_dict['model']['recon_decoders.0.module_1.0.weight'] = self.netG.state_dict()['recon_decoders.0.module_1.0.weight']
            self.netG.load_state_dict(state_dict['model'])
        
        self.optimizer_G = self._get_optim(self.netG.parameters())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        # self.netF = PWCNet(1).cuda()
        # state_dict = torch.load('./pretrained/pwcnet2')
        # self.netF.load_state_dict(state_dict)
        # self.netG = ReconFlowNet(c_in=32, scale=self.scale).cuda()
        # state_dict = torch.load('/home/ywqqqqqq/Documents/github/vidar_flow_recon/experiments/12_28_11_59_04/last_model.h5')
        # self.netG.load_state_dict(state_dict['model'])
        # self.netS = networks.VGG14_Spike().cuda()
        # for param in self.netS.parameters():
        #     param.requires_grad = False
        self.model = get_model(self.config['model'])
        # self.optimizer_S = optim.Adam(self.netS.parameters(), lr=self.config['optimizer']['lr']/10)
        # self.scheduler_S = LinearDecay(self.optimizer_S,
        #                             min_lr=self.config['scheduler']['min_lr']/10,
        #                             num_epochs=self.config['num_epochs'],
        #                             start_epoch=self.config['scheduler']['start_epoch'])

def main(config_path='config/config.yaml', mode='train', batch_size=None, exp=None, lambda_recon=1, lambda_epe=1, lambda_tv=0.1, lambda_pm=10, lambda_cl=0.1, load_model=None):
    from datasets.spike_dataset import SpikeFlowGT,SpikeFlowLoader, SpikeFlowLoader_val
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # load config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # config['lambda_recon'] = 1
    # config['lambda_epe'] = 1
    # config['lambda_tv'] = 0.1
    # config['lambda_pm'] = 10
    # config['lambda_cl'] = 0
    config['lambda_recon'] = 1
    config['lambda_epe'] = 1
    config['lambda_tv'] = 0.1
    config['lambda_pm'] = 10
    config['lambda_cl'] = 0
    # setting for loading data
    if batch_size is None:
        batch_size = config.pop('batch_size')
    device = config.pop('device')
    #train_data_path = config.pop('train')['data_path']
    #val_data_path = config.pop('val')['data_path']
    
    train_data_path = '/ssd_datasets/zl/experiment/big_object_flow/'#'/ssd_datasets/zl/experiment/43/' 
    val_data_path = '/ssd_datasets/zl/experiment/opticaltrain/'#'/ssd_datasets/zl/experiment/43/' 
    if mode == 'train':
        train_data_path = os.path.join(train_data_path, 'train.npz')
        val_data_path = os.path.join(val_data_path, 'val.npz')
    else:
        train_data_path = os.path.join(train_data_path, 'debug.npz')
        val_data_path = os.path.join(val_data_path, 'val.npz')

    # get datasets
    train_dataset = SpikeFlowGT(train_data_path, 'train')
    train_loader = SpikeFlowLoader(train_dataset, device, 1, 'train')

    val_dataset = SpikeFlowGT(val_data_path, 'val')
    val_loader = SpikeFlowLoader_val(val_dataset, device, 1, 'val')

    # train
    trainer = Trainer(config, train=train_loader, val=val_loader, exp=exp, load_model=load_model)
    trainer.train()

if __name__ == '__main__':
    Fire(main)