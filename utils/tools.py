import numpy as np
import torch
from math import ceil, floor
from torch.nn import ZeroPad2d

def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size

class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)
        
def backwarp(tenInput, tenFlow, backwarp_tenGrid = {}, backwarp_tenPartial = {}):
    #plt.imshow(tenInput.detach().cpu().numpy()[0].transpose(1,2,0), cmap='gray')
    #plt.show()
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy[image_numpy<-1] = -1
    image_numpy[image_numpy>1] = 1
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def seq_backwarp(x, flow):
    batch_size, time_step, height, width = x.shape
    x_align = torch.zeros_like(x)
    x_align[:,0] = x[:,0]
    unit_flow = flow / (time_step-1)
    for t in range(1,time_step):
        x_align[:,t:t+1] = backwarp(x[:,t:t+1], unit_flow*t)
    return x_align

def TFP(spk):
    return spk.sum(1,keepdims=True) / spk.sum(1).max()

def TFI(spk:np.ndarray,
        rc:int = 64, 
        height:int = 256,
        width:int = 400,
        flip:bool = False, 
        # pixel_table:list = [0, 255, 127, 85, 64, 51, 43, 36, 32, 28, 26, 23,
        #           21, 20, 18, 17, 16, 15, 14, 13, 13, 12, 12, 11,
        #           11, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7,
        #           6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        #           4, 4, 4, 4, 4, 4, 4]):
    
        pixel_table:list = [0, 255, 186, 155, 136, 123, 113, 105, 99, 94,
                 90, 86, 82, 79, 77, 74, 72, 70, 69, 67, 65, 64,
                 63, 61, 60, 59, 58, 57, 56, 55, 54, 54, 53, 52,
                 51, 51, 50, 49, 49, 48, 48, 47, 47, 46, 46, 45,
                 45, 44, 44, 43, 43, 43, 42, 42, 42, 41, 41, 41, 40,
                 40, 40, 39, 39, 39]):

    spk = spk[:rc]
    cur_spk = spk[rc//2-1] # center of spike sequence

    # get nearest fired spike from current time step in forward order
    exist = {} # whether spike exists at location (x,y)
    forward = np.concatenate([np.zeros((1,height,width)), spk[rc//2:rc]], 0)
    forward_first = np.zeros((height, width))
    for i in np.transpose(np.nonzero(forward)):
        if (i[1],i[2]) not in exist:
            forward_first[i[1],i[2]] = i[0]
            exist[(i[1],i[2])]=1
    
    # get nearest fired spike from current time step in backward order
    exist = {}
    backward = np.concatenate([np.zeros((1,height,width)), np.flipud(spk[:rc//2-1])], 0)
    backward_first = np.zeros(spk.shape[1:])
    for i in np.transpose(np.nonzero(backward)):
        if (i[1],i[2]) not in exist:
            backward_first[i[1],i[2]] = i[0]
            exist[(i[1],i[2])]=1

    # get pixel value by interval of spike
    # if spike fires at current time step, interval = time step of nearest fired_spike - current time step
    # if spike does not fire at current time step, interval = time step of nearest fired_spike (forward) - time step of nearest fired_spike (backward)
    # ps. if only one spike fired at all time step for any pixel, the pixel value is 0 which is pixel_table[0]. The minimum interval should be 0.
    img = np.zeros((height, width))
    img += (forward_first)*cur_spk 
    temp = (1-cur_spk)*(forward_first>0.5)*(backward_first>0.5)*(forward_first+backward_first) 
    
    img += temp
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]=pixel_table[int(img[i,j])]
    if flip:
        img = np.flip(img, 1)
    return img

def TFI1(spk, rc):
    pixel_table = [ 46,  47,  47,  48,  48,  49,  49,  50,  51,  51,  52,  53,  54,
        54,  55,  56,  57,  58,  59,  60,  61,  63,  64,  65,  67,  69,
        70,  72,  74,  77,  79,  82,  86,  90,  94,  99, 105, 113, 123,
       136, 155, 186, 255, 255]

    
    spk11 = spk
    b,c,h,w = spk.shape
    img_all = torch.zeros([b,1,h,w]).to('cuda')
    spk = spk.permute(0,2,3,1)
    for i in range(b):
        img = torch.zeros([h,w]).to('cuda')
        spk_i = spk[i,:]
        
        cur_spk = spk_i[:,:,rc]
        
        spk1 = spk_i[:,:,:rc]
        spk2 = spk_i[:,:,rc:]
        
        spkmap1 = torch.nonzero(spk1==1)
        spkmap1 = torch.flip(spkmap1,[0]) 
        spkmap2 = torch.nonzero(spk2==1)
        
        exist = {} 
        forward = torch.zeros([h,w]).to('cuda')
        backward = torch.zeros([h,w]).to('cuda')
        for j in spkmap1:
            if (j[0],j[1]) not in exist:
                forward[j[0],j[1]] = rc - j[2]
                exist[(j[0],j[1])]=1
        
        exist = {} 
        for j in spkmap2:
            if (j[0],j[1]) not in exist:
                backward[j[0],j[1]] = j[2]
                exist[(j[0],j[1])]=1    
                
        img += (forward)*cur_spk 
        temp = (1-cur_spk)*(forward>0.5)*(backward>0.5)*(forward+backward) 
        
        img += temp          
          
        for i1 in range(img.shape[0]):
            for j1 in range(img.shape[1]):
                img[i1,j1]=pixel_table[int(img[i1,j1])]      
        #cv.imwrite('/home/zhulin/reconflow/code_0205/vidar_flow_recon/experiments/img_new_sz.png', img.cpu().numpy())
        img_all[i,0,:,:] = img
    return img_all
