import torch
import torch.autograd as autograd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils.image_pool import ImagePool
import torch.nn.functional as F
from models.flownet import PWCNet
import numpy as np

from utils.tools import backwarp
###############################################################################
# Functions
###############################################################################

def gaussian_kernel(kernel_size = 7, sig = 1.0):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    center = kernel_size//2, 
    x_axis = np.linspace(0, kernel_size-1, kernel_size) - center
    y_axis = np.linspace(0, kernel_size-1, kernel_size) - center
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig)) / (np.sqrt(2*np.pi)*sig)
    return kernel

class ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.vgg = self.contentFunc()
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            if i == 0:
                layer.weight = nn.Parameter(layer.weight.sum(1, keepdims=True))
                layer.in_channels = 1
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        for name, param in model.named_parameters():
            param.requires_grad = False
        return model
    def forward(self, frame_gt, recons):
        loss = 0
        scale = len(recons)-1
        for l in range(scale):
            gt = F.interpolate(frame_gt, scale_factor=2**(l-scale), mode='bilinear', align_corners=False)
            recon = recons[l]
            loss = loss + 2**(l-scale)*(0.5*self.loss_fn(gt, recon) + 0.005*self.loss_fn(self.vgg(gt), self.vgg(recon)))
        else:
            gt = frame_gt
            recon = recons[-1]
            loss = loss + 0.5*self.loss_fn(gt, recon) + 0.005*self.loss_fn(self.vgg(gt), self.vgg(recon))
        # down_scale = 1/(2**(len(recons_f)-1))
        # for recon_f, recon_b in zip(recons_f,recons_b):
        #     frame_f_gt = F.interpolate(frame_gt[:,0:1],mode='bilinear', scale_factor=down_scale)
        #     frame_b_gt = F.interpolate(frame_gt[:,1:2],mode='bilinear', scale_factor=down_scale)
        #     loss = loss + 0.5*self.loss_fn(frame_f_gt, recon_f)*down_scale + \
        #            0.005*self.loss_fn(self.vgg(frame_f_gt.repeat(1,3,1,1)), self.vgg(recon_f.repeat(1,3,1,1)))
        #     loss = loss + 0.5*self.loss_fn(frame_b_gt, recon_b)*down_scale + \
        #            0.005*self.loss_fn(self.vgg(frame_b_gt.repeat(1,3,1,1)), self.vgg(recon_b.repeat(1,3,1,1)))
        #     down_scale *= 2
        return loss 

class ReconLoss1(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.vgg = self.contentFunc()
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            if i == 0:
                layer.weight = nn.Parameter(layer.weight.sum(1, keepdims=True))
                layer.in_channels = 1
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        for name, param in model.named_parameters():
            param.requires_grad = False
        return model
    def forward(self, frame_gt, recons):
        loss = 0
        scale = len(recons)-1

        gt = frame_gt
        recon = recons
        loss = loss + 0.5*self.loss_fn(gt, recon) + 0.005*self.loss_fn(self.vgg(gt), self.vgg(recon))
        # down_scale = 1/(2**(len(recons_f)-1))
        # for recon_f, recon_b in zip(recons_f,recons_b):
        #     frame_f_gt = F.interpolate(frame_gt[:,0:1],mode='bilinear', scale_factor=down_scale)
        #     frame_b_gt = F.interpolate(frame_gt[:,1:2],mode='bilinear', scale_factor=down_scale)
        #     loss = loss + 0.5*self.loss_fn(frame_f_gt, recon_f)*down_scale + \
        #            0.005*self.loss_fn(self.vgg(frame_f_gt.repeat(1,3,1,1)), self.vgg(recon_f.repeat(1,3,1,1)))
        #     loss = loss + 0.5*self.loss_fn(frame_b_gt, recon_b)*down_scale + \
        #            0.005*self.loss_fn(self.vgg(frame_b_gt.repeat(1,3,1,1)), self.vgg(recon_b.repeat(1,3,1,1)))
        #     down_scale *= 2
        return loss 
    
class EPELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
    def forward(self, flow_gt, flows):
        loss = 0
        scale = len(flows)-1
        for l in range(scale):
            gt = F.interpolate(flow_gt, scale_factor=2**(l-scale), mode='bilinear', align_corners=False)
            flow = flows[l]
            loss = loss + 2**(l-scale)*self.loss_fn(gt, flow)
        else:
            gt = flow_gt
            flow = flows[-1]
            loss = loss + self.loss_fn(gt, flow)
        # loss = self.loss_fn(flow_f_gt, flows_f)
        # loss = loss+self.loss_fn(flow_b_gt, flows_b)
        # loss = 0
        # down_scale = 1/(2**(len(flows_f)-1))
        # for flow_f, flow_b in zip(flows_f,flows_b):
        #     flow_f_gt_s = F.interpolate(flow_f_gt[:,0:1],mode='bilinear', scale_factor=down_scale)
        #     flow_b_gt_s = F.interpolate(flow_b_gt[:,1:2],mode='bilinear', scale_factor=down_scale)
        #     loss = loss + self.loss_fn(flow_f_gt_s, flow_f)*down_scale
        #     loss = loss + self.loss_fn(flow_b_gt_s, flow_b)*down_scale
        #     down_scale *= 2
        return loss

class TVLoss(nn.Module):
    def __init__(self, beta=2):
        super().__init__()
        self.beta = beta
    def forward(self, img):
        batch_size, channels, height, width = img.shape
        count_h = (height-1)*width*channels
        count_w = (width-1)*height*channels
        h_tv = ((img[:,:,1:,:]-img[:,:,:-1,:])**2).sum()
        w_tv = ((img[:,:,:,1:]-img[:,:,:,:-1])**2).sum()
        loss = self.beta*(h_tv/count_h+w_tv/count_w)/batch_size
        return loss
        
class PhotoMetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_coef = 20
    
    def forward(self, recon_f, recon_b, flow_f, flow_b, occ_f=None, occ_b=None):
        B,C,H,W = recon_f.shape
        recon_f = recon_f.detach()
        recon_b = recon_b.detach()
        
        w_reocn_f = backwarp(recon_b, flow_f*self.flow_coef)
        w_reocn_b = backwarp(recon_f, flow_b*self.flow_coef)

        if occ_f is None:
            pm_loss_f = ((w_reocn_f-recon_f)**2).mean()
        else:
            pm_loss_f = (((w_reocn_f-recon_f)**2) * (1-occ_f)).sum([1,2,3]) / (C*(H*W-occ_f.sum([1,2,3])))
            pm_loss_f = pm_loss_f.mean()

        if occ_b is None:
            pm_loss_b = ((w_reocn_b-recon_b)**2).mean()
        else:
            pm_loss_b = (((w_reocn_b-recon_b)**2) * (1-occ_b)).sum([1,2,3]) / (C*(H*W-occ_b.sum([1,2,3])))
            pm_loss_b = pm_loss_b.mean()

        return pm_loss_f+pm_loss_b

class ExposureLoss(nn.Module):
    def __init__(self, E=0.6, kernel_size=(8,8)):
        super().__init__()
        self.E = E
        self.kernel_size = kernel_size
    def forward(self, img):
        pooled_img = F.avg_pool2d(img, self.kernel_size)
        loss = (pooled_img-self.E).abs().mean()
        return loss

class SpacialConsLoss(nn.Module):
    def __init__(self, pool_kernel_size=(8,8)):
        super().__init__()
        kernel_left = torch.FloatTensor([[0,0,0],
                                              [-1,1,0],
                                              [0,0,0]])[None,None,...].cuda()
        kernel_right = torch.FloatTensor([[0,0,0],
                                              [0,1,-1],
                                              [0,0,0]])[None,None,...].cuda()
        kernel_top = torch.FloatTensor([[0,-1,0],
                                              [0,1,0],
                                              [0,0,0]])[None,None,...].cuda()
        kernel_down = torch.FloatTensor([[0,0,0],
                                              [0,1,0],
                                              [0,-1,0]])[None,None,...].cuda()        

        self.kernel = nn.Parameter(torch.cat((kernel_left,kernel_right,kernel_top,kernel_down)), requires_grad=False)
        # self.kernel_left = nn.Parameter(kernel_left, requires_grad=False)
        # self.kernel_right = nn.Parameter(kernel_right, requires_grad=False)
        # self.kernel_top = nn.Parameter(kernel_top, requires_grad=False)                              
        # self.kernel_down = nn.Parameter(kernel_down, requires_grad=False)

        self.pool_kernel_size = pool_kernel_size
    def forward(self, x):
        x = F.conv2d(x, self.kernel)
        # pooled_outputs = F.avg_pool2d(outputs, self.pool_kernel_size)
        # pooled_tfp = F.avg_pool2d(tfp, self.pool_kernel_size)

        # left_pooled_outputs = F.pad(pooled_outputs, (1,1,1,1))
        # left_pooled_tfp = F.pad(pooled_tfp, (1,1,1,1))
        # D_left_outputs = F.conv2d(left_pooled_outputs, self.kernel_left)
        # D_left_tfp = F.conv2d(left_pooled_tfp, self.kernel_left)
        # D_left = torch.pow(D_left_outputs - D_left_tfp, 2)

        # right_pooled_outputs = F.pad(pooled_outputs, (1,1,1,1))
        # right_pooled_tfp = F.pad(pooled_tfp, (1,1,1,1))
        # D_right_outputs = F.conv2d(right_pooled_outputs, self.kernel_right)
        # D_right_tfp = F.conv2d(right_pooled_tfp, self.kernel_right)
        # D_right = torch.pow(D_right_outputs - D_right_tfp, 2)

        # top_pooled_outputs = F.pad(pooled_outputs, (1,1,1,1))
        # top_pooled_tfp = F.pad(pooled_tfp, (1,1,1,1))
        # D_top_outputs = F.conv2d(top_pooled_outputs, self.kernel_top)
        # D_top_tfp = F.conv2d(top_pooled_tfp, self.kernel_top)
        # D_top = torch.pow(D_top_outputs - D_top_tfp, 2)

        # down_pooled_outputs = F.pad(pooled_outputs, (1,1,1,1))
        # down_pooled_tfp = F.pad(pooled_tfp, (1,1,1,1))
        # D_down_outputs = F.conv2d(down_pooled_outputs, self.kernel_down)
        # D_down_tfp = F.conv2d(down_pooled_tfp, self.kernel_down)
        # D_down = torch.pow(D_down_outputs - D_down_tfp, 2)

        return x

class PatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, patch1, patch2):
        loss = 0
        num_layers = len(patch1)
        for p1, p2 in zip(patch1, patch2):
            B,P,C = p1.shape
            p1 = F.normalize(p1, p=2, dim=-1)
            p2 = F.normalize(p2, p=2, dim=-1)
            logit = torch.bmm(p1, p2.transpose(1,2)).view(B*P,P)
            target = torch.arange(P, device=p1.device).unsqueeze(0).repeat(B,1).view(B*P)
            loss = loss + self.loss(logit, target) / num_layers
        return loss
    

class L_spa(nn.Module):
    def __init__(self, pool_kernel_size):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        # self.pool = nn.AvgPool2d(pool_kernel_size,stride=1)
        self.gaussian = torch.FloatTensor(gaussian_kernel(pool_kernel_size, pool_kernel_size/4.0)).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

    def forward(self, enhance, org):
        b, c, h, w = org.shape
        
        # # ### gray version
        # # org_mean = torch.mean(org, 1, keepdim=True)
        # # enhance_mean = torch.mean(enhance, 1, keepdim=True)
        # ### color version
        org_pool = F.conv2d(org, self.gaussian, padding=0, groups=3)
        enhance_pool = F.conv2d(enhance, self.gaussian, padding=0, groups=3)
        
        # weight_diff = torch.max(
        #     torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
        #                                                       torch.FloatTensor([0]).cuda()),
        #     torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)
        
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=0, groups=3)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=0, groups=3)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=0, groups=3)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=0, groups=3)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=0, groups=3)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=0, groups=3)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=0, groups=3)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=0, groups=3)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down

        return E

class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)

    
class MyPerceptualLoss():
    
    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()
        self.flowFunc = self.flowFunc()
        self.dyn_weight = 1
        
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        for name, param in model.named_parameters():
            param.requires_grad = False
        return model
    def flowFunc(self):
        model = PWCNet().cuda()
        state_dict = torch.load('./pretrained/pwcnet')
        for k,v in state_dict.items():
            state_dict[k] = torch.sum(v, 1, keepdim=True)
            break
        model.load_state_dict(state_dict)
        for name, param in model.named_parameters():
            param.requires_grad = False
        return model


    def get_gan_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm.repeat(1,3,1,1))
        f_real = self.contentFunc.forward(realIm.repeat(1,3,1,1))
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

    def get_loss(self, fakeIm, tfi, tfp):

            
        f_fake = self.contentFunc.forward(fakeIm.repeat(1,3,1,1))
        f_tfi = self.contentFunc.forward(tfi[:,:1].repeat(1,3,1,1))
        f_tfp = self.contentFunc.forward(tfp.repeat(1,3,1,1))
        mean_lum = F.avg_pool2d((tfi[:,:1]+1)*0.5,(4,4))
        # mean_lum = F.conv2d(mean_lum, self.kernel, padding=1)
        flow_forward = self.flowFunc.forward(tfi[:,1:2], tfi[:,2:3])
        flow_backward = self.flowFunc.forward(tfi[:,2:3], tfi[:,1:2])
        flow_forward = flow_forward.abs().sum(1,keepdims=True)
        flow_backward = flow_backward.abs().sum(1,keepdims=True)
        flow_union = torch.cat([flow_forward, flow_backward], 1).max(1, keepdims=True)[0].clamp(0,1)
        tfp_coef = (mean_lum+self.dyn_weight*(1-flow_union)**2)/(self.dyn_weight+1)
        tfi_coef = 1-tfp_coef
        f_real_no_grad = (tfp_coef*f_tfp+tfi_coef*f_tfi).detach()
        loss = self.criterion(f_fake, f_real_no_grad)

        tfp_coef = F.interpolate(tfp_coef, mode='nearest', scale_factor=4)
        tfi_coef = F.interpolate(tfi_coef, mode='nearest', scale_factor=4)
        return 0.006 * loss.mean() + 0.5 * F.mse_loss(fakeIm, (tfp_coef*tfp+tfi_coef*tfi[:,:1]).detach())
    def __call__(self, fakeIm, tfi, tfp):
        return self.get_loss(fakeIm, tfi, tfp)

class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        for name, param in model.named_parameters():
            param.requires_grad = False
        return model

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss
            self.contentFunc = self.contentFunc()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_loss(self, sim_outputs = None, sim_targets = None, real_outputs = None, f_spike = None):
        if sim_outputs is not None and sim_targets is not None:
            sim_outputs = (sim_outputs + 1) / 2.0
            sim_targets = (sim_targets + 1) / 2.0
            f_sim_outputs = self.contentFunc.forward(sim_outputs.repeat(1,3,1,1))
            f_sim_targets = self.contentFunc.forward(sim_targets.repeat(1,3,1,1)).detach()
            loss_sim = 0.006*self.criterion(f_sim_outputs, f_sim_targets) + 0.5*self.criterion(sim_outputs, sim_targets)
        else:
            loss_sim = 0
        
        if real_outputs is not None and f_spike is not None:
            real_outputs = (real_outputs + 1) / 2.0
            f_real_outputs = self.contentFunc.forward(real_outputs.repeat(1,3,1,1))
            loss_real = self.criterion(f_real_outputs, f_spike.detach())
        else:
            loss_real = 0
        
        if sim_outputs is None and sim_targets is not None:
            sim_targets = (sim_targets + 1) / 2.0
            f_sim_targets = self.contentFunc.forward(sim_targets.repeat(1,3,1,1)).detach()
            loss_sim = 0.006*self.criterion(f_real_outputs, f_sim_targets) + 0.5*self.criterion(real_outputs, sim_targets)
        
        return loss_sim, loss_real

    def __call__(self, sim_outputs = None, sim_targets = None, real_outputs = None, f_spike = None):
        return self.get_loss(sim_outputs, sim_targets, real_outputs, f_spike)
'''
class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss
            self.contentFunc = self.contentFunc()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_loss(self, fakeIm, realIm, f_spike = None):
        
        realIm = (realIm + 1) / 2.0
        # for i in range(fakeIm.shape[0]):
        #     fakeIm[i, :, :, :] = self.transform(fakeIm[i, :, :, :])
        #     realIm[i, :, :, :] = self.transform(realIm[i, :, :, :])
        f_real = self.contentFunc.forward(realIm.repeat(1,3,1,1))
        f_real_no_grad = f_real.detach()
        if fakeIm is not None:
            fakeIm = (fakeIm + 1) / 2.0
            f_fake = self.contentFunc.forward(fakeIm.repeat(1,3,1,1))
            mse_loss = self.criterion(f_fake, f_real_no_grad)
            loss = 0.006 * torch.mean(mse_loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)
        else:
            loss = torch.zeros(1)
        if f_spike is not None:
            loss_spike = self.criterion(f_spike, f_real_no_grad)
        else:
            loss_spike = torch.zeros(1)
        return loss, loss_spike

    def __call__(self, fakeIm, realIm, f_spike = None):
        return self.get_loss(fakeIm, realIm, f_spike)
'''
class GANLoss(nn.Module):
    def __init__(self, use_l2=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l2:
            self.loss = nn.MSELoss()#nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DiscLoss(nn.Module):
    def name(self):
        return 'DiscLoss'

    def __init__(self):
        super(DiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l2=True) ##
        self.fake_AB_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, True)

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLoss(nn.Module):
    def name(self):
        return 'RelativisticDiscLoss'

    def __init__(self):
        super(RelativisticDiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)
        errG = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 0) +
                self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 1)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 1) +
                       self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 0)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLossLS(nn.Module):
    def name(self):
        return 'RelativisticDiscLossLS'

    def __init__(self):
        super(RelativisticDiscLossLS, self).__init__()

        self.criterionGAN = GANLoss(use_l1=True)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.pred_fake = net.forward(fakeB)

        errG = torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) - 1) ** 2)
        # Real
        #self.pred_real = net.forward(realB)
        #errG = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) + 1) ** 2) +
        #        torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) - 1) ** 2)) / 2
    
        return errG

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2) +
                       torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def __init__(self):
        super(DiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)

    def get_g_loss(self, net, fakeB, realB):
        return DiscLoss.get_g_loss(self, net, fakeB)

    def get_loss(self, net, fakeB, realB):
        return DiscLoss.get_loss(self, net, fakeB, realB)


class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty


def get_loss(model):
    if model['content_loss'] == 'perceptual':
        content_loss = PerceptualLoss()
        content_loss.initialize(nn.MSELoss())
    elif model['content_loss'] == 'l1':
        content_loss = ContentLoss()
        content_loss.initialize(nn.L1Loss())
    elif model['content_loss'] == 'myperceptual':
        content_loss = MyPerceptualLoss(nn.MSELoss())
    else:
        raise ValueError("ContentLoss [%s] not recognized." % model['content_loss'])

    if model['disc_loss'] == 'wgan-gp':
        disc_loss = DiscLossWGANGP()
    elif model['disc_loss'] == 'lsgan':
        disc_loss = DiscLossLS()
    elif model['disc_loss'] == 'gan':
        disc_loss = DiscLoss()
    elif model['disc_loss'] == 'ragan':
        disc_loss = RelativisticDiscLoss()
    elif model['disc_loss'] == 'ragan-ls':
        disc_loss = RelativisticDiscLossLS()
    else:
        raise ValueError("GAN Loss [%s] not recognized." % model['disc_loss'])
    return content_loss, disc_loss
