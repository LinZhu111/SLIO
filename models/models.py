from flowiz import flowiz
import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM
from utils.metrics import PSNR
import flowiz as fz
import cv2
import lpips

class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

        return image_numpy.astype(imtype)

    def flow2im(self, flow_tensor, imtype=np.uint8):
        flow_numpy = flow_tensor[0].cpu().numpy()
        flow_numpy = flow_numpy.transpose(1,2,0)
        flow_numpy = fz.convert_from_flow(flow_numpy)
        return flow_numpy.astype(imtype)

    def gray2rgb(self, image):
        if image.shape[2] == 1:
            image = image.repeat(3,2)
        return image
    
    def get_images_and_metrics(self, recon_f, recon_b, flow_f, flow_b, frame_f = None, frame_b = None, flow_f_gt = None, flow_b_gt = None):
        recon_f = self.gray2rgb(self.tensor2im(recon_f.data))
        recon_b = self.gray2rgb(self.tensor2im(recon_b.data))
        flow_f = self.flow2im(flow_f.data)
        flow_b = self.flow2im(flow_b.data)
        
        # recon_f = self.flow2im(flow_f.data)
        # recon_b = self.flow2im(flow_b.data)
        # flow_f = recon_f
        # flow_b = recon_b

        if frame_f is not None and frame_b is not None:
            frame_f = self.gray2rgb(self.tensor2im(frame_f.data))
            frame_b = self.gray2rgb(self.tensor2im(frame_b.data))

            flow_f_gt = self.flow2im(flow_f_gt.data)
            flow_b_gt = self.flow2im(flow_b_gt.data)

            psnr = (PSNR(recon_f, frame_f) + PSNR(recon_b, frame_b))/2
            ssim = (SSIM(recon_f, frame_f, multichannel=True) + SSIM(recon_b, frame_b, multichannel=True))/2
        else:
            psnr = 0
            ssim = 0
        
        vis_img_pd = np.hstack((recon_f,recon_b,flow_f,flow_b))
        if frame_f is not None:
            vis_img_gt = np.hstack((frame_f,frame_b,flow_f_gt,flow_b_gt))
            vis_img = np.vstack((vis_img_pd,vis_img_gt))
        else:
            vis_img = vis_img_pd
        return psnr, ssim, vis_img

    def get_images_and_metrics2(self, recon_f, recon_b, flow_f, flow_b, frame_f = None, frame_b = None, flow_f_gt = None, flow_b_gt = None):
        recon_f = self.gray2rgb(self.tensor2im(recon_f.data))
        recon_b = self.gray2rgb(self.tensor2im(recon_b.data))
        flow_f = self.flow2im(flow_f.data)
        flow_b = self.flow2im(flow_b.data)
        
        # recon_f = self.flow2im(flow_f.data)
        # recon_b = self.flow2im(flow_b.data)
        # flow_f = recon_f
        # flow_b = recon_b

        if frame_f is not None and frame_b is not None:
            frame_f = self.gray2rgb(self.tensor2im(frame_f.data))
            frame_b = self.gray2rgb(self.tensor2im(frame_b.data))

            flow_f_gt = self.flow2im(flow_f_gt.data)
            flow_b_gt = self.flow2im(flow_b_gt.data)

            psnr = (PSNR(recon_f, frame_f) + PSNR(recon_b, frame_b))/2
            ssim = (SSIM(recon_f, frame_f, multichannel=True) + SSIM(recon_b, frame_b, multichannel=True))/2
        else:
            psnr = 0
            ssim = 0
        
        vis_img_pd = np.hstack((recon_f,recon_b,flow_f,flow_b))
        if frame_f is not None:
            vis_img_gt = np.hstack((frame_f,frame_b,flow_f_gt,flow_b_gt))
            vis_img = np.vstack((vis_img_pd,vis_img_gt))
        else:
            vis_img = vis_img_pd
        return psnr, ssim, vis_img
    
    def spike2dat(self, path, spike_in):
        width, height = 448, 256
        spike_out = np.flip(spike_in, 1)
        length = spike_out.shape[0]
        spike_out = spike_out.astype(np.bool).transpose(2,1,0)
        spike_out = np.array([spike_out[i] << (i & 7) for i in range(width)])
        spike_out = np.array([np.sum(spike_out[i: min(i+8, width)], axis=0)
                        for i in range(0, width, 8)]).astype(np.uint8)
        #spike_out = spike_out.reshape(fnum, height, width)
        file = open(path, "wb")
        for k1 in range(10):
            for i in range(spike_out.shape[2]):
                for j in range(spike_out.shape[1]):
                    for k in range(spike_out.shape[0]):
                        file.write(spike_out[k,j,i])
        file.close()
    
    def get_images(self, spike, frame_gt, flow_f_gt, flow_b_gt, occ_f, occ_b, path):
        self.spike2dat(path+'1.dat', spike[0].cpu().numpy())
        frame_gt = self.gray2rgb(self.tensor2im(frame_gt[:,:,0,:,:]))
        flow_f_gt = self.flow2im(flow_f_gt.data)
        flow_b_gt = self.flow2im(flow_b_gt.data)
        occ_f = self.gray2rgb(self.tensor2im(occ_f))
        occ_b = self.gray2rgb(self.tensor2im(occ_b))
        cv2.imwrite('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/check_dataset/frame_gt.bmp',frame_gt)
        cv2.imwrite('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/check_dataset/flow_f_gt.bmp',flow_f_gt)
        cv2.imwrite('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/check_dataset/flow_b_gt.bmp',flow_b_gt)
        cv2.imwrite('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/check_dataset/occ_f.bmp',occ_f)
        cv2.imwrite('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/check_dataset/occ_b.bmp',occ_b)
        
    
    def get_images_and_metrics1(self, recon_f, recon_b, flow_f, flow_b, frame_f = None, frame_b = None, flow_f_gt = None, flow_b_gt = None):
        recon_f = self.gray2rgb(self.tensor2im(recon_f.data))
        recon_b = self.gray2rgb(self.tensor2im(recon_b.data))
        flow_f = recon_f
        flow_b = recon_f
        
        # recon_f = self.flow2im(flow_f.data)
        # recon_b = self.flow2im(flow_b.data)
        # flow_f = recon_f
        # flow_b = recon_b

        if frame_f is not None and frame_b is not None:
            frame_f = self.gray2rgb(self.tensor2im(frame_f.data))
            frame_b = self.gray2rgb(self.tensor2im(frame_b.data))

            flow_f_gt = self.flow2im(flow_f_gt.data)
            flow_b_gt = self.flow2im(flow_b_gt.data)

            psnr = (PSNR(recon_f, frame_f) + PSNR(recon_b, frame_b))/2
            ssim = (SSIM(recon_f, frame_f, multichannel=True) + SSIM(recon_b, frame_b, multichannel=True))/2
        else:
            psnr = 0
            ssim = 0
        
        vis_img_pd = np.hstack((recon_f,recon_b,flow_f,flow_b))
        if frame_f is not None:
            vis_img_gt = np.hstack((frame_f,frame_b,flow_f_gt,flow_b_gt))
            vis_img = np.vstack((vis_img_pd,vis_img_gt))
        else:
            vis_img = vis_img_pd
        return psnr, ssim, vis_img

def get_model(model_config):
    return DeblurModel()
