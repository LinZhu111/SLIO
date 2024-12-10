import numpy as np
from multiprocessing import Process
from queue import Queue
from threading import Thread
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

        if partition=='train':
            self.spike = correct_data(self.spike, 2697, 3888, 3937)
            self.frame = correct_data(self.frame, 2697, 3888, 3937)
            self.flow_f = correct_data(self.flow_f, 2697, 3888, 3937)
            self.flow_b = correct_data(self.flow_b, 2697, 3888, 3937)
            self.occ_f = correct_data(self.occ_f, 2697, 3888, 3937)
            self.occ_b = correct_data(self.occ_b, 2697, 3888, 3937)
        
        self.len = len(self.spike)
        del data
        print(partition+' data loaded successfully')
    def __getitem__(self, index: int):
        return self.spike[index], self.frame[index], self.flow_f[index], self.flow_b[index], self.occ_f[index], self.occ_b[index]
    
    def __len__(self) -> int:
        return self.len
    
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
    


val_data_path = '/ssd_datasets/zl/reconflow/val.npz'
val_dataset = SpikeFlowGT(val_data_path, 'val')
val_loader = SpikeFlowLoader(val_dataset, 'cuda', 1, 'val')

for i, data in enumerate(val_loader):
    spike, frame_gt, flow_f_gt, flow_b_gt, occ_f, occ_b = data
    frame_f_gt, frame_b_gt = frame_gt[:,0], frame_gt[:,1]
    # 计算spike重构出的图片与frame_f_gt的PSNR SSIM
    # 计算spike重构出的图片与frame_b_gt的PSNR SSIM
    