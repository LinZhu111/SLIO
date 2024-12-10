#from albumentations.augmentations import transforms
import numpy as np
from threading import Thread
from multiprocessing import Process
from queue import Queue
#import albumentations as albu
from torch.utils import data
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import os
import time
from itertools import cycle, islice
import random

class QueueLoader():
    def __init__(self, dataset, device, batch_size=1, partition='train', queue_size=2):
        self.partition = partition
        if partition == 'train':
            self.loader = DataLoader(dataset, 
                                     batch_size, 
                                     shuffle=True, 
                                     drop_last=True)
        else:
            self.loader = DataLoader(dataset, 
                                     1, 
                                     shuffle=False, 
                                     drop_last=False)
        self.device = device
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.queue = Queue(maxsize=self.queue_size)
        
        self.idx = 0
        self.worker = Thread(target=self.preprocess)
        self.worker.setDaemon(True)
        self.worker.start()

    def preprocess(self):
        pass

    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        self.idx = 0
        return self
    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

class RealSpike(Dataset):
    def __init__(self, in_path, partition='train') -> None:
        super().__init__()
        self.partition = partition
        print('Loading real data')
        data = np.load(in_path)
        self.spike = data['spike']
        self.len = len(self.spike)

        if partition == 'train':
            self.clear = data['clear']
            self.len_c = len(self.clear)
            self.index_c = [i for i in range(self.len_c)]
            np.random.shuffle(self.index_c)
            self.i_c = 0
        print('Real data loaded successfully')

    def __getitem__(self, index: int):
        if self.partition == 'train':
            if self.i_c >= self.len_c-1:
                np.random.shuffle(self.index_c)
                self.i_c = 0
            clear = self.clear[[self.index_c[self.i_c], self.index_c[self.i_c+1]]]
            self.i_c += 2
            return self.spike[index], clear
        return self.spike[index]
    
    def __len__(self):
        return self.len

class RealSpikeLoader(QueueLoader):
    def __init__(self, dataset, device, batch_size=1, partition='train', queue_size=2):
        super().__init__(dataset, device, batch_size=batch_size, partition=partition, queue_size=queue_size)

    def decode(self, spike_temp, length=29):
        batch_size,_,height,width = spike_temp.shape
        spike = torch.zeros((batch_size,length,height,width))
        for i in range(length):
            spike[:,i:i+1] = spike_temp & 1
            spike_temp >>= 1
        return spike

    def crop(self, img, crop_size):
        B,_,H,W = img.shape

        h_pickable = [i for i in range(crop_size[0]//2, H-crop_size[0]//2)]
        w_pickable = [i for i in range(crop_size[1]//2, W-crop_size[1]//2)]

        loc_h = np.random.choice(h_pickable)
        loc_w = np.random.choice(w_pickable)

        c1 = (loc_h, loc_w) # center 1
        crop_img = img[:,:,(c1[0]-crop_size[0]//2):(c1[0]+crop_size[0]//2), (c1[1]-crop_size[1]//2):(c1[1]+crop_size[1]//2)]
        return crop_img

    def preprocess(self):
        while True:
            if self.partition == 'train':
                for spike, clear in self.loader:
                    B,C,H,W = spike.shape
                    spike = self.decode(spike)
                    
                    spike = F.pad(spike,[0,0,0,6],'constant',0)
                    spike = spike.float().to(self.device)
                    clear_f = clear[:,0]
                    clear_b = clear[:,1]
                    
                    crop_clear_f = self.crop(clear_f, (H,W))
                    crop_clear_f = (crop_clear_f.float()/128.0-1).to(self.device)

                    crop_clear_b = self.crop(clear_b, (H,W))
                    crop_clear_b = (crop_clear_b.float()/128.0-1).to(self.device)

                    input = (spike,crop_clear_f, crop_clear_b)
                    self.queue.put(input)
            else:
                for spike in self.loader:
                    spike = self.decode(spike)
                    spike = F.pad(spike,[0,0,0,6],'constant',0)
                    spike = spike.float().to(self.device)
                    input = spike
                    self.queue.put(input)

def correct_data(data, idx1, idx2, idx3):
    data[idx1,:] = data[idx1+100,:]
    data[idx2,:] = data[idx2+100,:]
    data[idx3,:] = data[idx3+100,:]
    return data
class SpikeFlowGT(Dataset):
    def __init__(self, in_path, partition='train') -> None:
        super().__init__()
        print('loading '+partition+' data')
        data = np.load(in_path)
        self.spike = data['spike'] #self.spike = data['clear']
        self.frame = data['frame']
        self.flow_f = data['flow_f']
        self.flow_b = data['flow_b']
        self.occ_f = data['occ_f']
        self.occ_b = data['occ_b']

        # if partition=='train':
        #     self.spike = correct_data(self.spike, 2697, 3888, 3937)
        #     self.frame = correct_data(self.frame, 2697, 3888, 3937)
        #     self.flow_f = correct_data(self.flow_f, 2697, 3888, 3937)
        #     self.flow_b = correct_data(self.flow_b, 2697, 3888, 3937)
        #     self.occ_f = correct_data(self.occ_f, 2697, 3888, 3937)
        #     self.occ_b = correct_data(self.occ_b, 2697, 3888, 3937)
        
        self.len = len(self.spike)
        del data
        print(partition+' data loaded successfully')
    def __getitem__(self, index: int):
        return self.spike[index], self.frame[index], self.flow_f[index], self.flow_b[index], self.occ_f[index], self.occ_b[index]
    
    def __len__(self) -> int:
        return self.len

class SpikeFlowLoader(QueueLoader):
    def __init__(self, dataset, device, batch_size=1, partition='train', queue_size=2):
        super().__init__(dataset, device, batch_size=batch_size, partition=partition, queue_size=queue_size)

    def decode(self, spike_temp, length=43):
        batch_size,_,height,width = spike_temp.shape
        spike = torch.zeros((batch_size,length,height,width))
        for i in range(length):
            spike[:,i:i+1] = spike_temp & 1
            spike_temp >>= 1
        return spike

    def preprocess(self):
        while True:
            for spike, frame, flow_f, flow_b, occ_f, occ_b in self.loader:
                spike = self.decode(spike)
                spike = spike.float().to(self.device)
                frame = (frame.float()/128.0-1).to(self.device)
                flow_f = ((flow_f.float()-2**14)/64.0).to(self.device)
                flow_f /= 20.0
                flow_b = ((flow_b.float()-2**14)/64.0).to(self.device)
                flow_b /= 20.0
                occ_f = occ_f.float().to(self.device).unsqueeze(1)
                occ_b = occ_b.float().to(self.device).unsqueeze(1)
                input = (spike,frame,flow_f,flow_b,occ_f,occ_b)
                self.queue.put(input)

class SpikeFlowLoader_val(QueueLoader):
    def __init__(self, dataset, device, batch_size=1, partition='train', queue_size=2):
        super().__init__(dataset, device, batch_size=batch_size, partition=partition, queue_size=queue_size)

    def decode(self, spike_temp, length=105):
        batch_size,_,height,width = spike_temp.shape
        spike_temp1 = spike_temp[:,0,:,:]
        spike_temp2 = spike_temp[:,1,:,:]
        spike = torch.zeros((batch_size,length,height,width))
        for i in range(64):
            spike[:,i:i+1] = spike_temp1 & 1
            spike_temp1 >>= 1
        for i in range(41):
            spike[:,i+64:i+65] = spike_temp2 & 1
            spike_temp2 >>= 1
        return spike

    def preprocess(self):
        while True:
            for spike, frame, flow_f, flow_b, occ_f, occ_b in self.loader:
                spike = self.decode(spike)
                spike = spike.float().to(self.device)
                frame = (frame.float()/128.0-1).to(self.device)
                flow_f = ((flow_f.float()-2**14)/64.0).to(self.device)
                flow_f /= 20.0
                flow_b = ((flow_b.float()-2**14)/64.0).to(self.device)
                flow_b /= 20.0
                occ_f = occ_f.float().to(self.device).unsqueeze(1)
                occ_b = occ_b.float().to(self.device).unsqueeze(1)
                input = (spike,frame,flow_f,flow_b,occ_f,occ_b)
                self.queue.put(input)
                
class SpikeFlowGT_CL(Dataset):
    def __init__(self, in_path, in_path1, partition='train') -> None:
        super().__init__()
        print('loading '+partition+' data')
        data = np.load(in_path)
        self.spike = data['spike'] #self.spike = data['clear']
        self.frame = data['frame']
        self.flow_f = data['flow_f']
        self.flow_b = data['flow_b']
        self.occ_f = data['occ_f']
        self.occ_b = data['occ_b']
        
        self.len = len(self.spike)
        
        data1 = np.load(in_path1)
        self.spike_real = data1['spike']
        self.tfi = data1['tfi']
        self.tfp = data1['tfp']
        try:
            self.img_real = data1['clear']
        except:
            self.img_real = self.frame[:,0,:,:,:]
        
        self.len_real = len(self.spike_real)
        if self.len_real > self.len:
            self.spike_real = self.spike_real[:self.len]
        else:
            self.spike_real = np.array(list(islice(cycle(self.spike_real), self.len)))
        
        self.len_real1 = len(self.img_real)
        if self.len_real > self.len:
            self.img_real = self.img_real[:self.len]
        else:
            self.img_real = np.array(list(islice(cycle(self.img_real), self.len)))

        self.len_real1 = len(self.tfi)
        if self.len_real > self.len:
            self.tfi = self.tfi[:self.len]
        else:
            self.tfi = np.array(list(islice(cycle(self.tfi), self.len)))
            self.tfp = np.array(list(islice(cycle(self.tfp), self.len)))         
        del data, data1
        print(partition+' data loaded successfully')
    def __getitem__(self, index: int):
        return self.spike[index], self.frame[index], self.flow_f[index], self.flow_b[index], self.occ_f[index], self.occ_b[index], self.spike_real[index], self.img_real[index], self.tfi[index], self.tfp[index]

    
    def __len__(self) -> int:
        return self.len

class SpikeFlowLoader_CL(QueueLoader):
    def __init__(self, dataset, device, batch_size=1, partition='train', queue_size=2):
        super().__init__(dataset, device, batch_size=batch_size, partition=partition, queue_size=queue_size)

    def decode(self, spike_temp, length=43):
        batch_size,_,height,width = spike_temp.shape
        spike = torch.zeros((batch_size,length,height,width))
        for i in range(length):
            spike[:,i:i+1] = spike_temp & 1
            spike_temp >>= 1
        return spike

    def preprocess(self):
        while True:
            for spike, frame, flow_f, flow_b, occ_f, occ_b, spike_real, img_real, tfi, tfp in self.loader:
                spike = self.decode(spike)
                spike = spike.float().to(self.device)
                spike_real = self.decode(spike_real)
                spike_real = spike_real.float().to(self.device)
                frame = (frame.float()/128.0-1).to(self.device)
                tfi = (tfi.float()/255.0).to(self.device)
                tfp = (tfp.float()/255.0).to(self.device)
                img_real = (img_real.float()/128.0-1).to(self.device)
                flow_f = ((flow_f.float()-2**14)/64.0).to(self.device)
                flow_f /= 20.0
                flow_b = ((flow_b.float()-2**14)/64.0).to(self.device)
                flow_b /= 20.0
                occ_f = occ_f.float().to(self.device)
                occ_b = occ_b.float().to(self.device)
                input = (spike,frame,flow_f,flow_b,occ_f,occ_b,spike_real,img_real,tfi,tfp)
                self.queue.put(input)
                
class TFIGT(Dataset):
    def __init__(self, in_path):
        super().__init__()
        self.tfi = np.load(in_path)
        self.gt = self.tfi['gt']
        self.tfi = self.tfi['tfi']
        
    def __len__(self) -> int:
        return self.length
    def __getitem__(self, index: int):
        scene = index // (22*31)
        clip = (index - (22*31)*scene) // 22
        start = index - (22*31)*scene - 22*clip

        tfi = np.load(os.path.join(self.in_path, '{0:03d}'.format(scene), 'data_tfi.npy'))[22*clip+start:22*clip+start+33]

        # tfi = np.zeros((33,256,448))
        # gt = np.zeros((1,256,448))
        # for t in range(start,start+33):
        #     tfi[0] = Image.open(os.path.join(self.in_path, '{0:03d}'.format(scene), '{0:02d}'.format(clip), '{0:02d}.png'.format(t)))
        # gt[0] = Image.open(os.path.join(self.in_path, '{0:03d}'.format(scene), '{0:02d}'.format(clip), 'gt', '{0:02d}.png'.format(start+16)))

        return tfi

class RealSpikeDataset(Dataset):
    def __init__(self, real_data_path, clear_data_path, partition='train', transform = None, corrupt = None) -> None:
        super().__init__()
        self.partition = partition
        self.transform = transform
        self.corrupt = corrupt
        print('loading '+partition+' data...')
        print('loading real spike data...')
        self.spike = np.load(real_data_path)
        self.spike, self.tfi = self.spike['spike'], self.spike['tfi']

        print('loading clear data...')
        self.gt = np.load(clear_data_path)
        self.gt = self.gt['gt'][:,:,:250,:400]
        print('loading '+partition+' data successfully')
        self.clear_index = [i for i in range(len(self.gt))]
        np.random.shuffle(self.clear_index)
        self.clear_i = 0

    def __len__(self):
        return len(self.spike)
    def __getitem__(self, index: int):
        if self.partition == 'train':
            spike = self.spike[index].transpose(1,2,0)
            if self.clear_i == len(self.clear_index):
                np.random.shuffle(self.clear_index)
                self.clear_i = 0
            gt = self.gt[self.clear_i].transpose(1,2,0)
            self.clear_i += 1
            tfi = self.tfi[index].transpose(1,2,0)
            if self.transform is not None:
                spike,gt,tfi = self.transform(spike,gt,tfi)
            spike_org = spike.transpose(2,0,1)
            if self.corrupt is not None:
                spike = self.corrupt(spike)
            spike = spike.transpose(2,0,1)
            gt = gt.transpose(2,0,1)
            tfi = tfi.transpose(2,0,1)
        else:
            spike = self.spike[index][:, :248]
            if self.clear_i == len(self.clear_index):
                np.random.shuffle(self.clear_index)
                self.clear_i = 0
            gt = self.gt[self.clear_i][:, :248]
            tfi = self.tfi[index][:, :248]
            spike_org = spike
        return spike, gt, tfi, spike_org

class RealSpikeDataLoader():
    def __init__(self, real_dataset, fake_dataset, device, batch_size=1, queue_size=2):

        self.real_loader = DataLoader(real_dataset, 
                                    batch_size, 
                                    shuffle=True, 
                                    drop_last=True)
        self.fake_loader = DataLoader(fake_dataset, 
                                    batch_size, 
                                    shuffle=True, 
                                    drop_last=True)
        self.device = device
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.queue = Queue(maxsize=self.queue_size)
        
        self.idx = 0
        self.worker = Thread(target=self.preprocess)
        self.worker.setDaemon(True)
        self.worker.start()

    def decode(self, spike_temp):
        batch_size,_,height,width = spike_temp.shape
        spike = torch.zeros((batch_size,32,height,width))
        for i in range(32):
            spike[:,i:i+1] = spike_temp & 1
            spike_temp >>= 1
        return spike

    def preprocess(self):
        while True:
            for (real_spike, clear, real_tfi, real_spike_org), (fake_spike, fake_gt, fake_tfi, fake_spike_org) in zip(self.real_loader, self.fake_loader):
                real_spike = self.decode(real_spike).to(self.device)
                fake_spike = self.decode(fake_spike).to(self.device)

                clear = clear.float().to(self.device)/128-1
                fake_gt = fake_gt.float().to(self.device)/128-1
                real_tfi = real_tfi.float().to(self.device)/128-1

                real_spike_org = self.decode(real_spike_org).to(self.device)
                real_tfp = real_spike_org.sum(1, keepdims=True)
                real_tfp = real_tfp/real_tfp.max()*2-1
                input = (real_spike,real_tfi,real_tfp,fake_spike,fake_gt,clear)
                self.queue.put(input)
                # self.queue.put((spike,gt,tfi))

    def __len__(self):
        return len(self.real_loader)
    def __iter__(self):
        self.idx = 0
        return self
    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.real_loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

class SpikeDataset(Dataset):
    def __init__(self, data_path, partition='train', transform = None, corrupt = None, load_gt=True, load_tfi=True) -> None:
        super().__init__()
        self.data_path = data_path
        self.partition = partition
        self.transform = transform
        self.corrupt = corrupt
        self.load_gt = load_gt
        self.load_tfi = load_tfi
        
        print('loading '+partition+' data...')
        self.spike = np.load(self.data_path)
        if load_gt:
            if partition == 'train':
                self.gt = self.spike['gt']
            else:
                self.gt = self.spike['gt'][:100]
        if load_tfi:
            if partition == 'train':
                self.tfi = self.spike['tfi']
            else:
                self.tfi = self.spike['tfi'][:100]
        if partition == 'train':
            self.spike = self.spike['spike']
        else:
            self.spike = self.spike['spike'][:100]
        
        print('loading '+partition+' data successfully')

    def __len__(self):
        return len(self.spike)

    def __getitem__(self, index: int):
        gt = None
        tfi = None
        if self.partition == 'train':
            spike = self.spike[index].transpose(1,2,0)

            if self.load_gt:
                gt = self.gt[index].transpose(1,2,0)
            if self.load_tfi:
                tfi = self.tfi[index].transpose(1,2,0)
            if self.transform is not None:
                spike,gt,tfi = self.transform(spike,gt,tfi)
            spike_org = spike.transpose(2,0,1)
            if self.corrupt is not None:
                spike = self.corrupt(spike)
            spike = spike.transpose(2,0,1)

            if self.load_gt:
                gt = gt.transpose(2,0,1)
            if self.load_tfi:
                tfi = tfi.transpose(2,0,1)
            else:
                tfi = torch.zeros(gt.shape)
        else:
            spike = self.spike[index]
            if self.load_gt:
                gt = self.gt[index]
            if self.load_tfi:
                tfi = self.tfi[index]
            else:
                tfi = torch.zeros(gt.shape)
            spike_org = spike
        return spike, gt, tfi, spike_org

class SpikeDataLoader(QueueLoader):
    def __init__(self, dataset, device, batch_size=1, partition='train', queue_size=2):
        super().__init__(dataset, device, batch_size=batch_size, partition=partition, queue_size=queue_size)

    def decode(self, spike_temp):
        batch_size,_,height,width = spike_temp.shape
        spike = torch.zeros((batch_size,32,height,width))
        for i in range(32):
            spike[:,i:i+1] = spike_temp & 1
            spike_temp >>= 1
        return spike

    def preprocess(self):
        while True:
            for spike, gt, tfi, spike_org in self.loader:
                spike = self.decode(spike)
                spike = spike.float().to(self.device)
                if gt is not None:
                    gt = gt.float().to(self.device)/255*2-1
                if tfi is not None:
                    tfi = (tfi.float().to(self.device)/255)*2-1
                spike_org = self.decode(spike_org).to(self.device)
                tfp = spike_org.sum(1, keepdims=True)
                tfp = tfp/tfp.max()*2-1
                input = (spike,gt,tfi,tfp)
                self.queue.put(input)
                # self.queue.put((spike,gt,tfi))



class RealSpikeDataset(Dataset):
    def __init__(self, real_data_path, clear_data_path, partition='train', transform = None, corrupt = None) -> None:
        super().__init__()
        self.partition = partition
        self.transform = transform
        self.corrupt = corrupt
        print('loading '+partition+' data...')
        print('loading real spike data...')
        self.spike = np.load(real_data_path)
        self.spike, self.tfi = self.spike['spike'], self.spike['tfi']

        print('loading clear data...')
        self.gt = np.load(clear_data_path)
        self.gt = self.gt['gt'][:,:,:250,:400]
        print('loading '+partition+' data successfully')
        self.clear_index = [i for i in range(len(self.gt))]
        np.random.shuffle(self.clear_index)
        self.clear_i = 0

    def __len__(self):
        return len(self.spike)
    def __getitem__(self, index: int):
        if self.partition == 'train':
            spike = self.spike[index].transpose(1,2,0)
            if self.clear_i == len(self.clear_index):
                np.random.shuffle(self.clear_index)
                self.clear_i = 0
            gt = self.gt[self.clear_i].transpose(1,2,0)
            self.clear_i += 1
            tfi = self.tfi[index].transpose(1,2,0)
            if self.transform is not None:
                spike,gt,tfi = self.transform(spike,gt,tfi)
            spike_org = spike.transpose(2,0,1)
            if self.corrupt is not None:
                spike = self.corrupt(spike)
            spike = spike.transpose(2,0,1)
            gt = gt.transpose(2,0,1)
            tfi = tfi.transpose(2,0,1)
        else:
            spike = self.spike[index][:, :248]
            if self.clear_i == len(self.clear_index):
                np.random.shuffle(self.clear_index)
                self.clear_i = 0
            gt = self.gt[self.clear_i][:, :248]
            tfi = self.tfi[index][:, :248]
            spike_org = spike
        return spike, gt, tfi, spike_org

class RealSpikeDataLoader():
    def __init__(self, real_dataset, fake_dataset, device, batch_size=1, queue_size=2):

        self.real_loader = DataLoader(real_dataset, 
                                    batch_size, 
                                    shuffle=True, 
                                    drop_last=True)
        self.fake_loader = DataLoader(fake_dataset, 
                                    batch_size, 
                                    shuffle=True, 
                                    drop_last=True)
        self.device = device
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.queue = Queue(maxsize=self.queue_size)
        
        self.idx = 0
        self.worker = Thread(target=self.preprocess)
        self.worker.setDaemon(True)
        self.worker.start()

    def decode(self, spike_temp):
        batch_size,_,height,width = spike_temp.shape
        spike = torch.zeros((batch_size,32,height,width))
        for i in range(32):
            spike[:,i:i+1] = spike_temp & 1
            spike_temp >>= 1
        return spike

    def preprocess(self):
        while True:
            for (real_spike, clear, real_tfi, real_spike_org), (fake_spike, fake_gt, fake_tfi, fake_spike_org) in zip(self.real_loader, self.fake_loader):
                real_spike = self.decode(real_spike).to(self.device)
                fake_spike = self.decode(fake_spike).to(self.device)

                clear = clear.float().to(self.device)/128-1
                fake_gt = fake_gt.float().to(self.device)/128-1
                real_tfi = real_tfi.float().to(self.device)/128-1

                real_spike_org = self.decode(real_spike_org).to(self.device)
                real_tfp = real_spike_org.sum(1, keepdims=True)
                real_tfp = real_tfp/real_tfp.max()*2-1
                input = (real_spike,real_tfi,real_tfp,fake_spike,fake_gt,clear)
                self.queue.put(input)
                # self.queue.put((spike,gt,tfi))

    def __len__(self):
        return len(self.real_loader)
    def __iter__(self):
        self.idx = 0
        return self
    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.real_loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = RealSpike('/media/ywqqqqqq/文档/dataset/VidarCity/dataset/train/debug.npz', 'train')
    loader = RealSpikeLoader(dataset, 'cuda', 1, partition='train')
    for spike in loader:
        print()

