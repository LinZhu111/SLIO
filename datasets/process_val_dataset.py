import os
import numpy as np
from tqdm import tqdm
import multiprocessing

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

def decode(spike_temp):
    batch_size,_,height,width = spike_temp.shape
    spike = np.zeros((batch_size,32,height,width))
    for i in range(32):
        spike[:,i:i+1] = spike_temp & 1
        spike_temp >>= 1
    return spike

def decode1(spike_temp):
    _,height,width = spike_temp.shape
    spike = np.zeros((64,height,width))
    for i in range(64):
        spike[:,i:i+1] = spike_temp & 1
        spike_temp >>= 1
    return spike

def dat2npy(shuffle, categ, scenes):
    
    for scene in tqdm(scenes):
        dats = os.listdir(os.path.join(path,categ,scene))
        for dat in dats:
            spike = decode_from_dat(os.path.join(path,categ,scene,dat),(400,250,400))
            comp_spike = np.zeros((5,250,400), dtype=np.int32)
            for i in range(5):
                for j in range(32):
                    comp_spike[i] += spike[i*80+j]<<j
            shuffle.append(comp_spike)
            
def decode(spike_temp, length=105):
    _,height,width = spike_temp.shape
    spike = np.zeros((length,height,width))
    for i in range((length-1)//64+1):
        for j in range(min(64, length-i*64)):
            spike[i*64+j] = ((spike_temp[i]>>j) & 1)
    return spike

def dat2npy(path, scene):
    dats = os.listdir(os.path.join(path,scene))
    for dat in dats:
        spike = decode_from_dat(os.path.join(path,scene,dat),(400,250,400))
        comp_spike = np.zeros((5,250,400), dtype=np.int32)
        for i in range(5):
            for j in range(32):
                comp_spike[i] += spike[i*80+j]<<j
                    
if __name__ == '__main__':
    spike = decode_from_dat('/home/zhulin/reconflow/code_0205/vidar_flow_recon/real_data/rotation1.dat',(400,250,400))
    comp_spike = np.zeros((1,250,400), dtype=np.int32)
    for i in range(1):
        for j in range(32):
            comp_spike[i] += spike[i*64+j]<<j    
    
    a = decode1(comp_spike)
    
    shuffle = multiprocessing.Manager().list()
    path = '/media/ywqqqqqq/YWQ/Dataset/VidarCity/Real/VidarCity/dataset/sampled/spike'
    categ1 = 'low motion'
    categ2 = 'motion'
    scenes1 = os.listdir(os.path.join(path, categ1))
    scenes2 = os.listdir(os.path.join(path, categ2))

    p1 = multiprocessing.Process(target = dat2npy, kwargs = {'shuffle':shuffle,
                                                        'categ':categ1,
                                                        'scenes':scenes1[:30]})
    p2 = multiprocessing.Process(target = dat2npy, kwargs = {'shuffle':shuffle,
                                                        'categ':categ1,
                                                        'scenes':scenes1[30:60]})
    p3 = multiprocessing.Process(target = dat2npy, kwargs = {'shuffle':shuffle,
                                                        'categ':categ1,
                                                        'scenes':scenes1[60:90]})
    p4 = multiprocessing.Process(target = dat2npy, kwargs = {'shuffle':shuffle,
                                                        'categ':categ1,
                                                        'scenes':scenes1[90:]})

    p5 = multiprocessing.Process(target = dat2npy, kwargs = {'shuffle':shuffle,
                                                        'categ':categ2,
                                                        'scenes':scenes2[:30]})
    p6 = multiprocessing.Process(target = dat2npy, kwargs = {'shuffle':shuffle,
                                                        'categ':categ2,
                                                        'scenes':scenes2[30:60]})
    p7 = multiprocessing.Process(target = dat2npy, kwargs = {'shuffle':shuffle,
                                                        'categ':categ2,
                                                        'scenes':scenes2[60:90]})
    p8 = multiprocessing.Process(target = dat2npy, kwargs = {'shuffle':shuffle,
                                                        'categ':categ2,
                                                        'scenes':scenes2[90:]})

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    shuffle_np = np.array(shuffle)
    shuffle_np = shuffle_np.reshape(-1,250,400)
    np.random.shuffle(shuffle_np)
    np.save('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Real/VidarCity/dataset/sampled/spike/train/train.npy', shuffle_np)
    np.save('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Real/VidarCity/dataset/sampled/spike/train/debug.npy', shuffle_np[0:12])


    print()