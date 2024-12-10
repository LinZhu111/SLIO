import argparse
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import models
import cv2
import os
import os.path as osp
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import inspect
from models.reconflownet_cl2 import ReconFlowNet_SNNencoder_dilation
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

import warnings
warnings.filterwarnings('ignore')

###############################################################################################################################################
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

    #spike_img_seq = np.flip(spike_img_seq, 1) # Vidar camera flips spike data vertically first
    return spike_img_seq

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = torch.nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = torch.nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


# def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
#     torch.save(state, os.path.join(save_path,filename))
#     if is_best:
#         shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

# class InputPadder:
#     """ Pads images such that dimensions are divisible by 16 """
#     def __init__(self, dims):
#         self.ht, self.wd = dims[-2:]
#         pad_ht = (((self.ht // 16) + 1) * 16 - self.ht) % 16
#         pad_wd = (((self.wd // 16) + 1) * 16 - self.wd) % 16
#         self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

#     def pad(self, *inputs):
#         return [F.pad(x, self._pad, mode='replicate') for x in inputs]

#     def unpad(self,x):
#         ht, wd = x.shape[-2:]
#         c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
#         return x[..., c[0]:c[1], c[2]:c[3]]


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def outflow_img(flow_list, vis_path, name_prefix='flow', max_batch=4):
    flow = flow_list[0]
    batch_size, c, h, w = flow.shape

    for batch in range(batch_size):
        if batch > max_batch:
            break
        flow_current = flow[batch,:,:,:].permute(1,2,0).detach().cpu().numpy()
        flow_img = flow_to_img_scflow(flow_current)

        cv2.imwrite(vis_path + '/{:s}_batch_id={:02d}.png'.format(name_prefix, batch), flow_img)
    
    return 

def out_img(img, vis_path, name_prefix=None, max_batch=4):
    batch_size, c, h, w = img.shape

    for batch in range(batch_size):
        if batch > max_batch:
            break
        
        img_current = img[batch, :, :, :].permute(1,2,0).detach().cpu().numpy() * 255.0
        cv2.imwrite(vis_path + '/{:s}_batch_id={:02d}.png'.format(name_prefix, batch), img_current)

    return

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors_scflow(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_img_scflow(flow_uv, clip_flow=None):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    convert_to_bgr = False
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors_scflow(u, v, convert_to_bgr)
# def writeFlow(filename,uv,v=None):
#     """ Write optical flow to file.
    
#     If v is None, uv is assumed to contain both u and v channels,
#     stacked in depth.
#     Original code by Deqing Sun, adapted from Daniel Scharstein.
#     """
#     nBands = 2

#     if v is None:
#         assert(uv.ndim == 3)
#         assert(uv.shape[2] == 2)
#         u = uv[:,:,0]
#         v = uv[:,:,1]
#     else:
#         u = uv

#     assert(u.shape == v.shape)
#     height,width = u.shape
#     f = open(filename,'wb')
#     # write the header
#     f.write(TAG_CHAR)
#     np.array(width).astype(np.int32).tofile(f)
#     np.array(height).astype(np.int32).tofile(f)
#     # arrange into matrix form
#     tmp = np.zeros((height, width*nBands))
#     tmp[:,np.arange(width)*2] = u
#     tmp[:,np.arange(width)*2 + 1] = v
#     tmp.astype(np.float32).tofile(f)
#     f.close()


# def supervised_loss(flow_preds, flow_gt, loss_dict):
#     w_scales = loss_dict['w_scales']
#     res_dict = {}
#     res_dict['flow_mean'] = flow_preds[0].abs().mean()
#     pym_losses = []
#     _, _, H, W = flow_gt.shape
    
#     for i, flow in enumerate(flow_preds):
#         b, c, h, w = flow.shape
#         flowgt_scaled = F.interpolate(flow_gt, (h, w), mode='bilinear') * (h / H)

#         curr_loss = (flowgt_scaled - flow).abs().mean()
#         pym_losses.append(curr_loss)
    
#     loss = [l * int(w) for l, w in zip(pym_losses, w_scales)]
#     loss = sum(loss)
#     return loss, res_dict


def compute_aee(flow_gt, flow_pred):
    EE = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    EE = torch.from_numpy(EE)

    if torch.sum(EE) == 0:
        AEE = 0
    else:
        AEE = torch.mean(EE)

    return  AEE

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser()
parser.add_argument('-tr', '--test_root', type=str, default='/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/phm/phm/test/', help='root path of test datasets')
parser.add_argument('-dt', '--dt', type=int, default=20, help='delta index between the input for flow')
parser.add_argument('-a', '--arch', default='scflow', choices=model_names, 
                    help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
parser.add_argument('-bn', '--batch_norm', default=False, type=bool, help='if use batch normlization during training')
parser.add_argument('-j', '--workers', default=1, type=int, help='number of data loading workers')
parser.add_argument('--pretrained', dest='pretrained', default=None, help='path to pre-trained model')
parser.add_argument('--print-detail', '-pd', action='store_true')
parser.add_argument('--eval_root', '-er', default='/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/phm/phm/')
args = parser.parse_args()

n_iter = 0
eval_vis_path = args.eval_root + '_dt{:d}'.format(args.dt)
if not osp.exists(eval_vis_path):
    os.makedirs(eval_vis_path)

class Test_loading1(Dataset):
    def __init__(self,  scene=None, transform=None):
        self.scene = scene
        self.samples, self.spike_dir = self.collect_samples()

    def collect_samples(self):
        scene_list = [self.scene]
        samples = []
        for scene in scene_list:
            spike_dir = osp.join(args.test_root, str(scene), '{:s}.dat'.format(str(scene)))
            flowgt_dir = osp.join(args.test_root, str(scene), 'dt={:d}'.format(args.dt), 'flow')
            for st in range(0, len(glob.glob(flowgt_dir+'/*.flo')) - 1):
                flow_path = flowgt_dir + '/{:04d}.flo'.format(int(st))
                if osp.exists(flow_path):
                    s = {}
                    s['flow_path'] = flow_path
                    samples.append(s)
        return samples, spike_dir

    def _load_sample(self, index, s):
        seq1 = decode_from_dat(self.spike_dir,(41,500,800),(s+1)*20-10)
        flow = readFlow(index['flow_path']).astype(np.float32)
        return seq1, flow

    def __len__(self):
        return len(self.samples)-1

    def __getitem__(self, index):
        seq1, flow = self._load_sample(self.samples[index+1], index)
        return seq1.copy(), flow.copy()

class Test_loading(Dataset):
    def __init__(self,  scene=None, transform=None):
        self.scene = scene
        self.samples, self.spike = self.collect_samples()

    def collect_samples(self):
        scene_list = [self.scene]
        samples = []
        for scene in scene_list:
            spike_dir = osp.join(args.test_root, str(scene), '{:s}.dat'.format(str(scene)))
            flowgt_dir = osp.join(args.test_root, str(scene), 'dt={:d}'.format(args.dt), 'flow')
            for st in range(0, len(glob.glob(flowgt_dir+'/*.flo')) - 1):
                flow_path = flowgt_dir + '/{:04d}.flo'.format(int(st))
                if osp.exists(flow_path):
                    s = {}
                    s['flow_path'] = flow_path
                    samples.append(s)
            seq1 = decode_from_dat(spike_dir,(100,500,800),0)
        return samples, seq1

    def _load_sample(self, index, s):
        seq1 = self.spike[(s+1)*20-10:(s+2)*20+13,:,:,:]
        flow = readFlow(index['flow_path']).astype(np.float32)
        return seq1, flow

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq1, flow = self._load_sample(self.samples[index], index)
        return seq1.copy(), flow.copy()
    
def validate(test_loader, model, scene):
    #model.eval()

    AEE_sum = 0.
    eval_time_sum = 0.
    iters = 0.
    scene_eval_vis_path = osp.join(eval_vis_path, scene)
    if not osp.exists(scene_eval_vis_path):
        os.makedirs(scene_eval_vis_path)
    log = open('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/phm/phm/log-best1.txt', mode = "a", encoding = "utf-8")
    
    for i, data in enumerate(test_loader, 0):
        seq1, flowgt_raw = data

        # compute output
        seq1 = seq1.cuda().type(torch.cuda.FloatTensor)
        flowgt = flowgt_raw.cuda().type(torch.cuda.FloatTensor).permute([0, 3, 1, 2])
        seq1 = seq1[:,:,:480,:]
        flowgt = flowgt[:,:,:480,:]
        padder = []#InputPadder(seq1_raw.shape)

        st_time = time.time()
        with torch.no_grad():
            flows_f_pd, flows_b_pd, recons_f_pd, recons_b_pd = model(seq1)
        eval_time = time.time() - st_time

        functional.reset_net(model)

        pred_flow = (flows_f_pd[-1]*20).detach().permute([0, 2, 3, 1]).squeeze().cpu().numpy()
        flowgt = flowgt.detach().permute([0, 2, 3, 1]).squeeze().cpu().numpy()

        flowgt_vis = flow_to_img_scflow(flowgt)
        pred_flow_vis = flow_to_img_scflow(pred_flow)
        pred_flow_vis_path = osp.join(scene_eval_vis_path, '{:03d}_pred.png'.format(i))
        cv2.imwrite(pred_flow_vis_path, pred_flow_vis)
        flowgt_vis_vis_path = osp.join(scene_eval_vis_path, '{:03d}_flowgt.png'.format(i))
        cv2.imwrite(flowgt_vis_vis_path, flowgt_vis)
        
        AEE = compute_aee(flowgt, pred_flow)
        
        AEE_sum += AEE
        eval_time_sum += eval_time

        iters += 1
        
        
        print('iters: {:04d}'.format(i))
        print('Scene: {:8s}, Index {:04d}, AEE: {:6.4f}, iters: {:04d}'.format(scene, i, AEE, int(iters)), file = log)
        print(AEE_sum/iters, file = log)

    # print('-------------------------------------------------------')
    print('Scene: {:s}, Mean AEE: {:6.4f}, Mean Eval Time: {:6.4f}'.format(scene, AEE_sum / iters, eval_time_sum / iters), file = log)
    print(AEE_sum/iters)
    print('-------------------------------------------------------')

    return AEE_sum / iters


def main():
    global args, best_EPE, image_resize, event_interval, spiking_ts, device, sp_threshold
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # create model
    model = ReconFlowNet_SNNencoder_dilation(c_in=32, scale=3).cuda()
    state_dict = torch.load('/home/zhulin/reconflow/code_0205/vidar_flow_recon/experiments/11_04_08_14_26train-flow-bs1/best1.h5') #11_01_11_21_32ReconFlowNet_SNNencoder_dilation-bs1-4411-final 
    model.load_state_dict(state_dict['model'])
    
    #model = torch.nn.DataParallel(model).cuda()
    #cudnn.benchmark = True
    
    # for scene in os.listdir(args.test_root):
    for scene in ['ball', 'cook', 'dice', 'dolldrop', 'fan', 'fly', 'hand', 'jump', 'poker', 'top']:
        Test_dataset = Test_loading1(scene=scene)
        test_loader = DataLoader(dataset=Test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
        EPE = validate(test_loader, model, scene)
    
if __name__ == '__main__':
    main()