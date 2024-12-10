import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.networks import get_generator
from models.flownet import PWCNet
from utils.tools import tensor2im, TFP, TFI
from models.reconflownet_cl2 import ReconFlowNet_SNNencoder_dilation
import flowiz as fz
import os
import torch.nn as nn
from PIL import Image
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch_flops import TorchFLOPsByFX
def plf_TFI(img_seq, rc=64, hsize=448, vsize=256, val_lut=[0, 255, 186, 155, 136, 123, 113, 105, 99, 94,
                 90, 86, 82, 79, 77, 74, 72, 70, 69, 67, 65, 64,
                 63, 61, 60, 59, 58, 57, 56, 55, 54, 54, 53, 52,
                 51, 51, 50, 49, 49, 48, 48, 47, 47, 46, 46, 45,
                 45, 44, 44, 43, 43, 43, 42, 42, 42, 41, 41, 41, 40,
                 40, 40, 39, 39, 39]):
    img = np.zeros((vsize, hsize), np.uint8)
    # print(fcnt)
    for i in range(vsize):
        for j in range(hsize):
            pix_seq = img_seq[:, i, j]
            pulse_forward = pix_seq[int(rc / 2):rc]
            back_tmp = pix_seq[0:int(rc / 2) - 1]
            pulse_backward = np.flipud(back_tmp)
            pulse_backward = np.append(pulse_backward, 0)  # for what?
            index0 = np.nonzero(pulse_backward)[0]
            index1 = np.nonzero(pulse_forward)[0]
            pulse_pix = int(pix_seq[int(rc / 2) - 1])
            if pulse_pix == 1:  # pulse_pix == 1时只往forward看？backward呢？
                if index1.size == 0:
                    res = 0
                else:
                    res = index1[0] + 1
            else:
                if index1.size == 0 or index0.size == 0:
                    res = 0
                else:
                    res = index1[0] + index0[0] + 2
            img[i, j] =  val_lut[int(res)]
    return img

def byte_to_spike(byte_seq:bytes):
    len_spike = len(byte_seq)*8
    spike_seq = [(byte_seq[i//8]>>(i%8)) & 1 for i in range(len_spike)]
    return spike_seq

def decode_from_dat(dat_path: str, size:tuple, frame_skip:int = 0):
    num_frame = size[0]
    height = size[1]
    width = size[2]
    len_bytes = num_frame*width*height//8
    with open(dat_path, 'rb') as dat_file:
        dat_file.seek(frame_skip*width*height//8)
        byte_seq = dat_file.read(len_bytes)
    spike_seq = byte_to_spike(byte_seq)
    spike_img_seq = np.resize(spike_seq, size)

    spike_img_seq = np.flip(spike_img_seq, 1) # Vidar camera flips spike data vertically first
    return spike_img_seq

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = ReconFlowNet_SNNencoder_dilation(c_in=32, scale=3).cuda()
    state_dict = torch.load('./best_model.h5') #experiments/04_14_09_12_21train43
    model.load_state_dict(state_dict['model'], strict=False)

    data = decode_from_dat('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/phm/fan.dat',(41,500,800),200)
    
    with torch.no_grad():
        spike = torch.FloatTensor(data[None,:43,:480].copy()).cuda()
        


        flows_f_pd, flows_b_pd, recons_f_pd, recons_b_pd = model(spike)

        flow_f = fz.convert_from_flow(flows_f_pd[-1][0].cpu().numpy().transpose(1,2,0))
        flow_b = fz.convert_from_flow(flows_b_pd[-1][0].cpu().numpy().transpose(1,2,0))

        recon_f = tensor2im(recons_f_pd[-1]).repeat(3,2)
        recon_b = tensor2im(recons_b_pd[-1]).repeat(3,2)

        tfp = tensor2im(TFP(spike)).repeat(3,2)
        tfi = plf_TFI(spike.cpu().numpy()[0],32, 400, 248)[:,:,None].repeat(3,2)
        
        img = Image.fromarray(tfp)
        img=img.convert("L")
        #img = img.rotate(180,expand=True)
        img.save('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/tfp.bmp')
        cv2.imwrite('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/recon_b11.bmp',recon_b)
        cv2.imwrite('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/flow_b11.bmp',flow_b)