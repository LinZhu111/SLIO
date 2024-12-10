import torch
import torch.nn as nn
import torch.nn.functional as F
if __name__ == '__main__':
    import correlation
else:
    from models import correlation
from model.snn_network import SNN_ConvLayer, SNN_ConvLayer_MP


FORWARD = 0
BACKWARD = 1
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



class BasicBlock(nn.Module):
    def __init__(self, features, norm=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.LeakyReLU(inplace=False, negative_slope=0.1)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.LeakyReLU(inplace=False, negative_slope=0.1)
        self.has_norm = (norm is not None)
        if self.has_norm:
            self.norm1 = norm(features)
            self.norm2 = norm(features)
    def forward(self, x):
        out = self.conv1(x)
        if self.has_norm:
            out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.has_norm:
            out = self.norm2(out)
        out = self.relu2(out)
        return x + out


class Encoder_snnlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm=None, tau=2.0, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Encoder_snnlayer, self).__init__()

        self.conv1 = SNN_ConvLayer_MP(in_channels, out_channels, kernel_size, 2, padding, norm, tau, v_threshold, v_reset, activation_type)
        self.conv2 = SNN_ConvLayer_MP(out_channels, out_channels, kernel_size, 1, padding, norm, tau, v_threshold, v_reset, activation_type)
        self.conv3 = SNN_ConvLayer_MP(out_channels, out_channels, kernel_size, 1, padding, norm, tau, v_threshold, v_reset, activation_type)
        #self.recurrent_block = rnn.SpikingConvLSTMCell(input_size=out_channels, hidden_size=out_channels)

    def forward(self, x):
        x, v, mp = self.conv1(x)
        x, v, mp = self.conv2(x)
        x, v, mp = self.conv3(x)
        return x, v, mp
    
class Encoder_snnlayer_sumV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm=None, tau=2.0, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Encoder_snnlayer_sumV, self).__init__()

        self.conv1 = SNN_ConvLayer_MP(in_channels, out_channels, kernel_size, 2, padding, norm, tau, v_threshold, v_reset, activation_type)
        self.conv2 = SNN_ConvLayer_MP(out_channels, out_channels, kernel_size, 1, padding, norm, tau, v_threshold, v_reset, activation_type)
        self.conv3 = SNN_ConvLayer_MP(out_channels, out_channels, kernel_size, 1, padding, norm, tau, v_threshold, v_reset, activation_type)
        #self.recurrent_block = rnn.SpikingConvLSTMCell(input_size=out_channels, hidden_size=out_channels)

    def forward(self, x):
        x, v1, mp1 = self.conv1(x)
        x, v2, mp2 = self.conv2(x)
        x, v3, mp3 = self.conv3(x)
        return x, v1+v2+v3, mp1+mp2+mp3
    
class Encoder(nn.Module):
    def __init__(self, scale=3, c_in=11, c_hid=[32, 64, 128], num_layer=5, feat_id=[0,2,4]):
        super().__init__()
        
        self.scale = scale
        self.num_layer = num_layer
        self.feat_id = feat_id
        self.module_1_spk = nn.Sequential(
                        nn.Conv2d(in_channels=c_in, out_channels=c_hid[0], kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=c_hid[0], out_channels=c_hid[0], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=c_hid[0], out_channels=c_hid[0], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.module_1_img = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=c_hid[0], kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=c_hid[0], out_channels=c_hid[0], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=c_hid[0], out_channels=c_hid[0], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.module_2 = nn.Sequential(
                        nn.Conv2d(in_channels=c_hid[0], out_channels=c_hid[1], kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=c_hid[1], out_channels=c_hid[1], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=c_hid[1], out_channels=c_hid[1], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.module_3 = nn.Sequential(
                        nn.Conv2d(in_channels=c_hid[1], out_channels=c_hid[2], kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=c_hid[2], out_channels=c_hid[2], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=c_hid[2], out_channels=c_hid[2], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
        self.module_4 = nn.Sequential(
                        nn.Conv2d(in_channels=c_hid[2]*5, out_channels=c_hid[2], kernel_size=3, stride=1, padding=1)
        )
        
        self.module_5 = nn.ModuleList()
        for i in range(num_layer):
            self.module_5.append(BasicBlock(c_hid[2])) #resblock
        self.module_1_spk_snn = Encoder_snnlayer(in_channels=c_in, out_channels=c_hid[0], kernel_size=3, stride=2, padding=1, norm=None, tau=2.0, v_threshold=1.0, v_reset=None, activation_type = 'lif')
        self.module_2_snn = Encoder_snnlayer(in_channels=c_hid[0], out_channels=c_hid[1], kernel_size=3, stride=2, padding=1, norm=None, tau=2.0, v_threshold=1.0, v_reset=None, activation_type = 'lif')
        self.module_3_snn = Encoder_snnlayer(in_channels=c_hid[1], out_channels=c_hid[2], kernel_size=3, stride=2, padding=1, norm=None, tau=2.0, v_threshold=1.0, v_reset=None, activation_type = 'lif')
                    
        self.module_1_spk_snn_sumV = Encoder_snnlayer_sumV(in_channels=c_in, out_channels=c_hid[0], kernel_size=3, stride=2, padding=1, norm=None, tau=2.0, v_threshold=1.0, v_reset=None, activation_type = 'plif')
        self.module_2_snn_sumV = Encoder_snnlayer_sumV(in_channels=c_hid[0], out_channels=c_hid[1], kernel_size=3, stride=2, padding=1, norm=None, tau=2.0, v_threshold=1.0, v_reset=None, activation_type = 'plif')
        self.module_3_snn_sumV = Encoder_snnlayer_sumV(in_channels=c_hid[1], out_channels=c_hid[2], kernel_size=3, stride=2, padding=1, norm=None, tau=2.0, v_threshold=1.0, v_reset=None, activation_type = 'plif')
        
    def forward_cl(self, x, in_type, batch_size=None):
        feat_cl= []
        if in_type == 'spike':
            x = self.module_1_spk(x)
            x = self.module_2(x)
            x = self.module_3(x)
            x_f = torch.cat(x.split(batch_size), 1)
            x_b = torch.cat(x.split(batch_size)[::-1], 1)
            x = torch.cat([x_f, x_b], 0)
            x = self.module_4(x)
        else:
            x = self.module_1_img(x)
            x = self.module_2(x)
            x = self.module_3(x)
        
        for i in range(self.num_layer):
            x = self.module_5[i](x)
            if i in self.feat_id:
                feat_cl.append(x)
        
        return feat_cl
        
    def forward_normal(self, x, batch_size):
        feat_enc = []
        x = self.module_1_spk(x)
        feat_enc.append(torch.cat(x[:,None].split(batch_size), 1))
        x = self.module_2(x)
        feat_enc.append(torch.cat(x[:,None].split(batch_size), 1))
        x = self.module_3(x)
        feat_enc.append(torch.cat(x[:,None].split(batch_size), 1))
        
        x_f = torch.cat(x.split(batch_size), 1)
        x_b = torch.cat(x.split(batch_size)[::-1], 1)
        x = torch.cat([x_f, x_b], 0)
        
        x = self.module_4(x)
        
        for i in range(self.num_layer):
            x = self.module_5[i](x)
        
        x_f = x[:batch_size]
        x_b = x[batch_size:]
        return x_f, x_b, feat_enc

    def forward_normal_SNN_mp(self, x, batch_size):
        x_in = x
        for iter in range(1):
            feat_enc = []
            x, v, mp = self.module_1_spk_snn(x_in)
            feat_enc.append(torch.cat(mp[:,None].split(batch_size), 1))
            x, v, mp = self.module_2_snn(x)
            feat_enc.append(torch.cat(mp[:,None].split(batch_size), 1))
            x, v, mp = self.module_3_snn(x)
            feat_enc.append(torch.cat(mp[:,None].split(batch_size), 1))
            
        x = mp
        x_f = torch.cat(x.split(batch_size), 1)
        x_b = torch.cat(x.split(batch_size)[::-1], 1)
        x = torch.cat([x_f, x_b], 0)
        
        x = self.module_4(x)
        
        for i in range(self.num_layer):
            x = self.module_5[i](x)
        
        x_f = x[:batch_size]
        x_b = x[batch_size:]
        return x_f, x_b, feat_enc

    def forward_normal_SNN_v(self, x, batch_size):
        x_in = x
        for iter in range(1):
            feat_enc = []
            x, v, mp = self.module_1_spk_snn(x_in)
            feat_enc.append(torch.cat(v[:,None].split(batch_size), 1))
            x, v, mp = self.module_2_snn(x)
            feat_enc.append(torch.cat(v[:,None].split(batch_size), 1))
            x, v, mp = self.module_3_snn(x)
            feat_enc.append(torch.cat(v[:,None].split(batch_size), 1))
            
        x = v
        x_f = torch.cat(x.split(batch_size), 1)
        x_b = torch.cat(x.split(batch_size)[::-1], 1)
        x = torch.cat([x_f, x_b], 0)
        
        x = self.module_4(x)
        
        for i in range(self.num_layer):
            x = self.module_5[i](x)
        
        x_f = x[:batch_size]
        x_b = x[batch_size:]
        return x_f, x_b, feat_enc

    def forward_normal_SNN_sumV(self, x, batch_size):
        x_in = x
        feat_enc =[]
        for iter in range(1):
            x, v, mp = self.module_1_spk_snn_sumV(x_in)
            if iter == 0:
                feat_enc1 = torch.zeros_like(torch.cat(v[:,None].split(batch_size), 1))
            feat_enc1 += torch.cat(v[:,None].split(batch_size), 1)
            x, v, mp = self.module_2_snn_sumV(x)
            if iter == 0:
                feat_enc2 = torch.zeros_like(torch.cat(v[:,None].split(batch_size), 1))
            feat_enc2 += torch.cat(v[:,None].split(batch_size), 1)
            x, v, mp = self.module_3_snn_sumV(x)
            if iter == 0:
                vsum = torch.zeros_like(v)
                feat_enc3 = torch.zeros_like(torch.cat(v[:,None].split(batch_size), 1))
            feat_enc3 += torch.cat(v[:,None].split(batch_size), 1)
            vsum += v
        
        feat_enc.append(feat_enc1)
        feat_enc.append(feat_enc2)
        feat_enc.append(feat_enc3)
        x = vsum
        x_f = torch.cat(x.split(batch_size), 1)
        x_b = torch.cat(x.split(batch_size)[::-1], 1)
        x = torch.cat([x_f, x_b], 0)
        
        x = self.module_4(x)
        
        for i in range(self.num_layer):
            x = self.module_5[i](x)
        
        x_f = x[:batch_size]
        x_b = x[batch_size:]
        return x_f, x_b, feat_enc

    def forward_normal_SNN_sumMP(self, x, batch_size):
        x_in = x
        feat_enc =[]
        for iter in range(8):
            x, v, mp = self.module_1_spk_snn_sumV(x_in)
            if iter == 0:
                feat_enc1 = torch.zeros_like(torch.cat(mp[:,None].split(batch_size), 1))
            feat_enc1 += torch.cat(mp[:,None].split(batch_size), 1)
            x, v, mp = self.module_2_snn_sumV(x)
            if iter == 0:
                feat_enc2 = torch.zeros_like(torch.cat(mp[:,None].split(batch_size), 1))
            feat_enc2 += torch.cat(mp[:,None].split(batch_size), 1)
            x, v, mp = self.module_3_snn_sumV(x)
            if iter == 0:
                vsum = torch.zeros_like(mp)
                feat_enc3 = torch.zeros_like(torch.cat(mp[:,None].split(batch_size), 1))
            feat_enc3 += torch.cat(mp[:,None].split(batch_size), 1)
            vsum += mp
        
        feat_enc.append(feat_enc1)
        feat_enc.append(feat_enc2)
        feat_enc.append(feat_enc3)
        x = vsum
        x_f = torch.cat(x.split(batch_size), 1)
        x_b = torch.cat(x.split(batch_size)[::-1], 1)
        x = torch.cat([x_f, x_b], 0)
        
        x = self.module_4(x)
        
        for i in range(self.num_layer):
            x = self.module_5[i](x)
        
        x_f = x[:batch_size]
        x_b = x[batch_size:]
        return x_f, x_b, feat_enc
        
    def forward_cl1(self, x, in_type, batch_size):
        if in_type == 'spike':
            feat_enc = []
            x = self.module_1_spk(x)
            b,_,h,w = x.shape
            feat_enc.append(torch.cat(x[:,None].split(batch_size), 1).view(int(b/5), -1, h, w))
            x = self.module_2(x)
            b,_,h,w = x.shape
            feat_enc.append(torch.cat(x[:,None].split(batch_size), 1).view(int(b/5), -1, h, w))
            x = self.module_3(x)
            b,_,h,w = x.shape
            feat_enc.append(torch.cat(x[:,None].split(batch_size), 1).view(int(b/5), -1, h, w))
        else:
            feat_enc = []
            x = self.module_1_img(x)
            feat_enc.append(x)
            x = self.module_2(x)
            feat_enc.append(x)
            x = self.module_3(x)
            feat_enc.append(x)            

        return feat_enc
        
    def forward(self, x, in_type='spike', step=8, length=11, mode='SNN'):
        batch_size, time_step, height, width = x.shape
        if in_type == 'spike':
            #batch_size, time_step, height, width = x.shape
            x_list = []
            i = 0
            while i*step+length <= time_step:
                x_list.append(x[:,i*step:i*step+length,:,:])
                i += 1
            x = torch.cat(x_list,0)

            
            # while i*step+length <= time_step:
            #     temp_x = torch.zeros_like(x[:,0:length,:,:])
            #     for j in range(length):
            #         temp = x[:,i*step:i*step+j+1,:,:]
            #         temp = torch.sum(temp,dim=1).unsqueeze(1)
            #         temp_x[:,j:j+1,:,:] = temp
            #     x_list.append(temp_x)
            #     i += 1
            # x = torch.cat(x_list,0)
                        
            if mode == 'cl':
                return self.forward_cl1(x, in_type, batch_size)
            elif mode == 'SNN':
                return self.forward_normal_SNN_sumV(x, batch_size)
            else:
                return self.forward_normal(x, batch_size)
            
        return self.forward_cl1(x, in_type, batch_size)

    def forward_normal_SNN_sumMP(self, x, batch_size):
        x_in = x
        feat_enc =[]
        for iter in range(8):
            x, v, mp = self.module_1_spk_snn_sumV(x_in)
            if iter == 0:
                feat_enc1 = torch.zeros_like(torch.cat(mp[:,None].split(batch_size), 1))
            feat_enc1 += torch.cat(mp[:,None].split(batch_size), 1)
            x, v, mp = self.module_2_snn_sumV(x)
            if iter == 0:
                feat_enc2 = torch.zeros_like(torch.cat(mp[:,None].split(batch_size), 1))
            feat_enc2 += torch.cat(mp[:,None].split(batch_size), 1)
            x, v, mp = self.module_3_snn_sumV(x)
            if iter == 0:
                vsum = torch.zeros_like(mp)
                feat_enc3 = torch.zeros_like(torch.cat(mp[:,None].split(batch_size), 1))
            feat_enc3 += torch.cat(mp[:,None].split(batch_size), 1)
            vsum += mp
        
        feat_enc.append(feat_enc1)
        feat_enc.append(feat_enc2)
        feat_enc.append(feat_enc3)
        x = vsum
        x_f = torch.cat(x.split(batch_size), 1)
        x_b = torch.cat(x.split(batch_size)[::-1], 1)
        x = torch.cat([x_f, x_b], 0)
        
        x = self.module_4(x)
        
        for i in range(self.num_layer):
            x = self.module_5[i](x)
        
        x_f = x[:batch_size]
        x_b = x[batch_size:]
        return x_f, x_b, feat_enc
        
    def forward_cl1(self, x, in_type, batch_size):
        if in_type == 'spike':
            feat_enc = []
            x = self.module_1_spk(x)
            b,_,h,w = x.shape
            feat_enc.append(torch.cat(x[:,None].split(batch_size), 1).view(int(b/5), -1, h, w))
            x = self.module_2(x)
            b,_,h,w = x.shape
            feat_enc.append(torch.cat(x[:,None].split(batch_size), 1).view(int(b/5), -1, h, w))
            x = self.module_3(x)
            b,_,h,w = x.shape
            feat_enc.append(torch.cat(x[:,None].split(batch_size), 1).view(int(b/5), -1, h, w))
        else:
            feat_enc = []
            x = self.module_1_img(x)
            feat_enc.append(x)
            x = self.module_2(x)
            feat_enc.append(x)
            x = self.module_3(x)
            feat_enc.append(x)            

        return feat_enc
        
    def forward(self, x, in_type='spike', step=8, length=11, mode='SNN'):
        batch_size, time_step, height, width = x.shape
        if in_type == 'spike':
            #batch_size, time_step, height, width = x.shape
            x_list = []
            i = 0
            while i*step+length <= time_step:
                x_list.append(x[:,i*step:i*step+length,:,:])
                i += 1
            x = torch.cat(x_list,0)

            
            # while i*step+length <= time_step:
            #     temp_x = torch.zeros_like(x[:,0:length,:,:])
            #     for j in range(length):
            #         temp = x[:,i*step:i*step+j+1,:,:]
            #         temp = torch.sum(temp,dim=1).unsqueeze(1)
            #         temp_x[:,j:j+1,:,:] = temp
            #     x_list.append(temp_x)
            #     i += 1
            # x = torch.cat(x_list,0)
                        
            if mode == 'cl':
                return self.forward_cl1(x, in_type, batch_size)
            elif mode == 'SNN':
                return self.forward_normal_SNN_sumV(x, batch_size)
            else:
                return self.forward_normal(x, batch_size)
            
        return self.forward_cl1(x, in_type, batch_size)
        
class FlowDecoder(nn.Module):
    def __init__(self, level):
        super().__init__()

        self.flow_coef = [2.5, 5, 10][level]

        c_in = [81+128, 81+96+2+2, 81+64+2+2][level]
        c_hid = [128,96,64,32,16] # [128,64,32,16,8]

        self.module_1 = nn.Sequential(nn.Conv2d(c_in, c_hid[0], kernel_size=3,stride=1,padding=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1))
        # self.module_2 = BasicBlock(c_hid)
        # self.module_3 = BasicBlock(c_hid)
        # self.module_4 = BasicBlock(c_hid)
        self.module_2 = nn.Sequential(nn.Conv2d(c_in+c_hid[0], c_hid[1], kernel_size=3,stride=1,padding=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.module_3 = nn.Sequential(nn.Conv2d(c_in+c_hid[0]+c_hid[1], c_hid[2], kernel_size=3,stride=1,padding=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.module_4 = nn.Sequential(nn.Conv2d(c_in+c_hid[0]+c_hid[1]+c_hid[2], c_hid[3], kernel_size=3,stride=1,padding=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.module_5 = nn.Sequential(nn.Conv2d(c_in+c_hid[0]+c_hid[1]+c_hid[2]+c_hid[3], c_hid[4], kernel_size=3,stride=1,padding=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.module_6 = nn.Conv2d(c_in+c_hid[0]+c_hid[1]+c_hid[2]+c_hid[3]+c_hid[4], 2, kernel_size=3, stride=1, padding=1)

        if level > 0:
            c_feat = [81+128, 81+96+2+2, 81+64+2+2][level-1]+128+96+64+32+16
            # self.module_upfeat = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
            #                                    nn.Conv2d(c_feat, 2, kernel_size=3, stride=1, padding=1))
            # self.module_upflow = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2))
            self.module_upfeat = nn.ConvTranspose2d(c_feat, 2, kernel_size=4, stride=2, padding=1)
            self.module_upflow = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
    # def calculate_cost_volume(self, x_1, x_2, flow=None):
    #     if flow is None:
    #         cost_volume = correlation.FunctionCorrelation(x_1, x_2)
    #     else:
    #         cost_volume = correlation.FunctionCorrelation(x_1, backwarp(x_2, flow))
    #     cost_volume = F.leaky_relu(cost_volume, negative_slope=0.1)
    #     return cost_volume

    def forward(self, x_1, x_2, feat_flow=None, flow=None): #x_1: x_2: feat_flow: feat_recon_f,feat_recon_b,feat_flow_f,
        
        if feat_flow is not None:
            feat_flow = self.module_upfeat(feat_flow)
            flow = self.module_upflow(flow)
            cost_vol = F.leaky_relu(correlation.FunctionCorrelation(x_1, backwarp(x_2,flow*self.flow_coef)), negative_slope=0.1)
            x = torch.cat([cost_vol, x_1, feat_flow, flow], 1)
        else:
            cost_vol = F.leaky_relu(correlation.FunctionCorrelation(x_1, x_2), negative_slope=0.1)
            x = torch.cat([cost_vol, x_1], 1)

        x = torch.cat([self.module_1(x), x], 1)
        x = torch.cat([self.module_2(x), x], 1)
        x = torch.cat([self.module_3(x), x], 1)
        x = torch.cat([self.module_4(x), x], 1)
        feat_flow = torch.cat([self.module_5(x), x], 1)
        flow = self.module_6(feat_flow)
        return feat_flow, flow

class ReconDecoder(nn.Module):
    def __init__(self, level):
        super().__init__()

        self.flow_coef = [2.5, 5, 10][level]
        
        c_in = [128+128,64+128,32+96][level]
        c_hid = [128,96,64][level]
        c_hid_prev = [128,96,64][level-1]
        self.module_up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                       nn.Conv2d(c_hid_prev, c_hid_prev, kernel_size=3, stride=1, padding=1)
                                    )
        # self.module_up = nn.Sequential(nn.ConvTranspose2d(c_hid_prev, c_hid_prev,kernel_size=4,stride=2,padding=1),
        #                                nn.LeakyReLU(negative_slope=0.1))
        self.module_1 = nn.Sequential(nn.Conv2d(c_in, c_hid, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1),)
        self.module_2 = BasicBlock(c_hid)
        self.module_3 = BasicBlock(c_hid)
        self.module_to_img = nn.Sequential(nn.Conv2d(c_hid, c_hid//2, kernel_size=3, stride=1, padding=1),
                                           nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                           nn.Conv2d(c_hid//2, 1, kernel_size=3, stride=1, padding=1),
                                           )
        
    def seq_backwarp(self, x, flow):
        batch_size, time_step, channel, height, width = x.shape
        x_align = torch.zeros_like(x)
        x_align[:,0] = x[:,0]
        unit_flow = flow / (time_step-1)
        for t in range(1,time_step):
            x_align[:,t] = backwarp(x[:,t], unit_flow*t)
#         x_align = x_align.reshape(batch_size, time_step*channel, height, width)
        x_mean = x_align.mean(1)
#         return x_align
        return x_mean

    def forward(self, feat_enc, flow=None, feat_recon=None):#enc_feat_f, flow=flow_f, feat_recon=feat_recon_f
        if flow is None:
            batch_size, time_step, channel, height, width = feat_enc.shape
            feat_enc = feat_enc.mean(1)
#             feat_enc = feat_enc.reshape(batch_size, time_step*channel, height, width)
#             feat_recon = feat_recon.reshape(batch_size, time_step*channel, height, width)
        else:
            flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False)
            feat_enc = self.seq_backwarp(feat_enc, self.flow_coef*flow)   #########
            feat_recon = self.module_up(feat_recon)
        x = torch.cat([feat_enc, feat_recon], 1)
        x = self.module_1(x)
        x = self.module_2(x)
        x = self.module_3(x)
        # x = self.module_up(x)
        feat_recon = x
        recon = self.module_to_img(x)
        return feat_recon, recon


    

class FlowRefiner_dilation(nn.Module):
    def __init__(self):
        super().__init__()
        c_in = 81+64+2+2+128+96+64+32+16
        self.module_1 = nn.Sequential(nn.Conv2d(c_in, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                      nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=8, dilation=8),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                      nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=16, dilation=16),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
                                      #nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                      )

    def forward(self, feat_flow):
        return self.module_1(feat_flow)
    

    
class ReconRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_coef = 10
        c_in = 64
        c_hid = 32
        c_hid_prev = 64
        self.module_up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                        nn.Conv2d(c_hid_prev, c_hid_prev, kernel_size=3, stride=1, padding=1)
                                    )
        # self.module_up = nn.Sequential(nn.ConvTranspose2d(c_hid_prev, c_hid_prev, kernel_size=4, stride=2, padding=1),
        #                                nn.LeakyReLU(negative_slope=0.1))
        self.module_1 = nn.Sequential(nn.Conv2d(c_in, c_hid, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.module_2 = BasicBlock(c_hid)
        self.module_3 = BasicBlock(c_hid)
        self.module_to_img = nn.Sequential(nn.Conv2d(c_hid, c_hid//2, kernel_size=3, stride=1, padding=1),
                                           nn.LeakyReLU(inplace=False, negative_slope=0.1),
                                           nn.Conv2d(c_hid//2, 1, kernel_size=3, stride=1, padding=1))
    
    def seq_backwarp(self, x, flow):
        batch_size, time_step, channel, height, width = x.shape
        x_align = torch.zeros_like(x)
        x_align[:,0] = x[:,0]
        unit_flow = flow / (time_step-1)
        for t in range(1,time_step):
            x_align[:,t] = backwarp(x[:,t], unit_flow*t)
#         x_align = x_align.reshape(batch_size, time_step*channel, height, width)
        x_mean = x_align.mean(1)
#         return x_align
        return x_mean

    def forward(self, spike, flow, feat_recon):
        # spike = spike[:,:,None]
        # x = self.seq_backwarp(spike, self.flow_coef*flow)
        
        feat_recon = self.module_up(feat_recon)
        # x = torch.cat([feat_recon, x], 1)
        x = feat_recon
        x = self.module_1(x)
        x = self.module_2(x)
        x = self.module_3(x)
        recon = self.module_to_img(x)
        return recon



class ReconFlowNet_SNNencoder_dilation(nn.Module): #default
    def __init__(self,  c_in=32, scale=3):
        super().__init__()
        self.c_in = c_in
        self.scale = scale

        self.encoder = Encoder(scale=3, c_in=11)

        self.flow_decoders = self.build_flow_decoder()

        self.recon_decoders = self.build_recon_decoder()
        
        self.flow_refiner = FlowRefiner_dilation()
        self.recon_refiner = ReconRefiner()


    def build_flow_decoder(self):
        flow_decoders = nn.ModuleList()
        for l in range(self.scale):
            flow_decoders.append(FlowDecoder(l))
        return flow_decoders
 
    def build_recon_decoder(self):
        recon_decoders = nn.ModuleList()
        for l in range(self.scale):
            recon_decoders.append(ReconDecoder(l))
        return recon_decoders

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, spike):
        flows_f = []
        flows_b = []
        recons_f = []
        recons_b = []
        tran_feats_enc_f = []
        tran_feats_enc_b = []
        feat_recon_f=None
        feat_recon_b=None
        flow_f=None
        flow_b=None
        feat_flow_f=None
        feat_flow_b=None
        

        feat_recon_f, feat_recon_b, enc_feats = self.encoder(spike, step=8, length=11) #8, 11  #5  21
#         feat_recon_b = feat_recon_f.flip(1)
        
#         batch_size, time_step, channel, height, width = feat_recon_f.shape
#         feat_recon_f = feat_recon_f.reshape(batch_size, time_step*channel, height, width)
#         feat_recon_b = feat_recon_b.reshape(batch_size, time_step*channel, height, width)
        
        for s in range(self.scale):
            enc_feat_f = enc_feats[-1-s]
            enc_feat_b = enc_feat_f.flip(1)
            
            feat_recon_f, recon_f = self.recon_decoders[s](enc_feat_f, flow=flow_f, feat_recon=feat_recon_f)
            feat_recon_b, recon_b = self.recon_decoders[s](enc_feat_b, flow=flow_b, feat_recon=feat_recon_b)
            
            feat_flow_f, flow_f = self.flow_decoders[s](feat_recon_f,feat_recon_b,feat_flow_f,flow_f)
            feat_flow_b, flow_b = self.flow_decoders[s](feat_recon_b,feat_recon_f,feat_flow_b,flow_b)

            flows_f.append(flow_f)
            flows_b.append(flow_b)
            recons_f.append(recon_f)
            recons_b.append(recon_b)

        else:
            spike_f = spike
            spike_b = spike.flip(1)
            flow_f = self.flow_refiner(feat_flow_f) + flow_f
            flow_b = self.flow_refiner(feat_flow_b) + flow_b

            flow_f = F.interpolate(flow_f, scale_factor=2, mode='bilinear', align_corners=False)
            flow_b = F.interpolate(flow_b, scale_factor=2, mode='bilinear', align_corners=False)

            recon_f = self.recon_refiner(spike_f, flow=flow_f, feat_recon=feat_recon_f)
            recon_b = self.recon_refiner(spike_b, flow=flow_b, feat_recon=feat_recon_b)

            flows_f.append(flow_f)
            flows_b.append(flow_b)
            recons_f.append(recon_f)
            recons_b.append(recon_b)
        return flows_f, flows_b, recons_f, recons_b
    
    

        

        

if __name__ == '__main__':
    spike = torch.zeros(1,43,256,448).cuda()
    image = torch.zeros(1,1,256,448).cuda()
    model = ReconFlowNet(43,3).cuda()
    flows_f, flows_b, recons_f, recons_b = model(spike)
    feats_img_f = model.encoder(image, 'image', mode='cl')
    feats_spk_f = model.encoder(spike, 'spike', mode='cl')
    print(len(feats_spk_f))
    print(len(feats_img_f))
    print(feats_spk_f[0].shape)
    print(1)