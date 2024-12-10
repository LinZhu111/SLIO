import logging
from collections import defaultdict
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

WINDOW_SIZE = 100


class MetricCounter:
    def __init__(self, exp_path):
        self.writer = SummaryWriter(exp_path)
        logging.basicConfig(filename=os.path.join(exp_path, 'exp.log'), level=logging.INFO)
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)
        self.best_metric = 0

    def add_image(self, x: np.ndarray, tag: str):
        self.images[tag].append(x)

    def clear(self):
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)

    def add_losses(self, l_G, l_recon=0, l_epe=0, l_pm=0, l_cl=0):
        for name, value in zip(('l_G', 'l_recon', 'l_epe', 'l_pm', 'l_cl'),
                               (l_G, l_recon, l_epe, l_pm, l_cl)):
            self.metrics[name].append(value)

    def add_metrics(self, psnr, ssim):
        for name, value in zip(('PSNR', 'SSIM'),
                               (psnr, ssim)):
            self.metrics[name].append(value)

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in ('l_G', 'l_recon', 'l_epe', 'l_pm', 'l_cl', 'PSNR', 'SSIM'))
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for tag in ('l_G', 'l_recon', 'l_epe', 'l_pm', 'l_cl', 'SSIM', 'PSNR'):
            self.writer.add_scalar(f'{scalar_prefix}_{tag}', np.mean(self.metrics[tag]), global_step=epoch_num)
        for tag in self.images:
            imgs = self.images[tag]
            if imgs:
                imgs = np.array(imgs)
                self.writer.add_images(tag, imgs[:, :, :, ::-1].astype('float32') / 255, dataformats='NHWC',
                                       global_step=epoch_num)
                self.images[tag] = []

    def update_best_model(self):
        cur_metric = np.mean(self.metrics['PSNR'])
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False

class MetricCounter_Real:
    def __init__(self, exp_path):
        self.writer = SummaryWriter(exp_path)
        logging.basicConfig(filename=os.path.join(exp_path, 'exp.log'), level=logging.INFO)
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)
        self.best_metric = 0

    def add_image(self, x: np.ndarray, tag: str):
        self.images[tag].append(x)

    def clear(self):
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)
        
    def add_metrics(self, psnr, ssim):
        for name, value in zip(('PSNR', 'SSIM'),
                               (psnr, ssim)):
            self.metrics[name].append(value)
            
    def add_losses(self, l_G, l_D, l_pm=0, l_cl=0, l_adv=0):
        for name, value in zip(('l_G', 'l_D', 'l_pm', 'l_cl', 'l_adv'),
                               (l_G, l_D, l_pm, l_cl, l_adv)):
            self.metrics[name].append(value)

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in ('l_G', 'l_D', 'l_pm', 'l_cl', 'l_adv', 'PSNR', 'SSIM'))
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for tag in ('l_G', 'l_D', 'l_pm', 'l_cl', 'l_adv', 'PSNR', 'SSIM'):
            self.writer.add_scalar(f'{scalar_prefix}_{tag}', np.mean(self.metrics[tag]), global_step=epoch_num)
        for tag in self.images:
            imgs = self.images[tag]
            if imgs:
                imgs = np.array(imgs)
                self.writer.add_images(tag, imgs[:, :, :, ::-1].astype('float32') / 255, dataformats='NHWC',
                                       global_step=epoch_num)
                self.images[tag] = []

    def update_best_model(self):
        cur_metric = -np.mean(self.metrics['l_G'])
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False
