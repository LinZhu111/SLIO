import numpy as np
import torch
from utils.tools import backwarp

def occlusion(flow_f, flow_b):
    B,_,H,W = flow_f.shape
    w_flow_b = backwarp(flow_b, flow_f)
    w_flow_f = backwarp(flow_f, flow_b)
    flow_diff_f = flow_f + w_flow_b
    flow_diff_b = flow_b + w_flow_f
    mag_sq_f = len_sq(flow_f) + len_sq(w_flow_b)
    mag_sq_b = len_sq(flow_b) + len_sq(w_flow_f)
    occ_thre_f = 0.01*mag_sq_f+0.5
    occ_thre_b = 0.01*mag_sq_b+0.5

    occ_f = len_sq(flow_diff_f)>occ_thre_f
    occ_b = len_sq(flow_diff_b)>occ_thre_b
    return occ_f, occ_b

def len_sq(x):
    return (x**2).sum(1,keepdims=True)

if __name__ == '__main__':
    data = np.load('/media/ywqqqqqq/文档/dataset/SpikeFlyingThings/train/debug.npz')
    flows_f = (data['flow_f'].astype(np.float32)- 2**15)/64.0
    flows_b = (data['flow_b'].astype(np.float32)- 2**15)/64.0

    occs_f = []
    occs_b = []
    for flow_f, flow_b in zip(flows_f, flows_b):
        occ_f, occ_b = occlusion(torch.FloatTensor(flow_f[None,...]).cuda(), torch.FloatTensor(flow_b[None,...]).cuda())
        occs_f.append(occ_f[0].cpu().numpy())
        occs_b.append(occ_b[0].cpu().numpy())
    occs_f = np.array(occs_f, dtype=bool)
    occs_b = np.array(occs_b, dtype=bool)

    flows_f = (flows_f*64.0 + 2**14).astype(np.int16)
    flows_b = (flows_b*64.0 + 2**14).astype(np.int16)
    np.savez('/media/ywqqqqqq/文档/dataset/SpikeFlyingThings/train/debug.npz', spike=data['spike'], 
                                                                            frame=data['frame'], 
                                                                            flow_f=data['flow_f'], 
                                                                            flow_b=data['flow_b'],
                                                                            occ_f=occs_f,
                                                                            occ_b=occs_b)    