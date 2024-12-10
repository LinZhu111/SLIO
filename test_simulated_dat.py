from models.reconflownet_cl2 import ReconFlowNet_SNNencoder_dilation
from utils.tools import TFI, TFP, tensor2im
from utils.metrics import SSIM, PSNR
from utils.niqe import niqe
import torch
import torch.nn as nn
from datasets.spike_dataset import SpikeFlowGT, SpikeFlowLoader
from tqdm import tqdm
import cv2
import numpy as np
import flowiz as fz

if __name__ == '__main__':
    model = ReconFlowNet_SNNencoder_dilation().cuda()

    state_dict = torch.load('/home/ywqqqqqq/Documents/github/vidar_flow_recon/experiments/01_14_12_44_31/last_model.h5')
    model.load_state_dict(state_dict['model'])

    sim_dataset = SpikeFlowGT('/media/ywqqqqqq/文档/dataset/SpikeFlyingThings/train/val_fl_occ.npz', 'val')
    sim_loader = SpikeFlowLoader(sim_dataset, 'cuda', 1, 'val')

    psnr = []
    ssim = []
    with torch.no_grad():
        for i, data in enumerate(sim_loader):
            spike, frame_gt, flow_f_gt, flow_b_gt,_,_ = data
            frame_f_gt, frame_b_gt = frame_gt[:,0], frame_gt[:,1]
            flows_f_pd, flows_b_pd, recons_f_pd, recons_b_pd = model(spike)

            flow_f = fz.convert_from_flow(flows_f_pd[-1][0].cpu().numpy().transpose(1,2,0))
            flow_b = fz.convert_from_flow(flows_b_pd[-1][0].cpu().numpy().transpose(1,2,0))

            flow_f_gt = fz.convert_from_flow(flow_f_gt[0].cpu().numpy().transpose(1,2,0))
            flow_b_gt = fz.convert_from_flow(flow_b_gt[0].cpu().numpy().transpose(1,2,0))

            recon_f = tensor2im(recons_f_pd[-1]).repeat(3,2)
            recon_b = tensor2im(recons_b_pd[-1]).repeat(3,2)

            recon_f[recon_f<0] = 0
            recon_f[recon_f>255] = 255
            recon_b[recon_b<0] = 0
            recon_b[recon_b>255] = 255

            tfp = tensor2im(TFP(spike)).repeat(3,2)
            tfi = TFI(spike.cpu().numpy()[0],32)[:,:,None].repeat(3,2)

            frame_f = tensor2im(frame_f_gt).repeat(3,2)
            frame_b = tensor2im(frame_b_gt).repeat(3,2)

            h1 = np.hstack([recon_f, recon_b, flow_f, flow_b])
            h2 = np.hstack([frame_f, frame_b, flow_f_gt, flow_b_gt])
            v = np.vstack([h1,h2])
            cv2.imwrite('/home/ywqqqqqq/Documents/github/vidar_flow_recon/result/val_rf_new/'+str(i)+'.jpg', v)