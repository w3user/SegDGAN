# --*-- coding:utf-8  --*--
from torch.utils.data import Dataset
import time

import cv2
import numpy as np
import torch
import torch.utils.data as data


class loader_npy(data.Dataset):
  
    def __init__(self, image_path, mask_path, mode='train'):
        super(loader_npy, self).__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.mode = mode

        self.data_imgs = np.load(self.image_path)
        self.data_masks = np.load(self.mask_path)

    def __getitem__(self, index):

        if self.mode == "train":
            return self.__getitem_train(index)
        elif self.mode == "val":
            return self.__getitem_val(index)
        elif self.mode == 'infer':
            return self.__getitem_infer(index)
        elif self.mode == 'test':
            return self.__getitem_test(index)

    def __getitem_train(self, index):
        np.random.seed(int(time.time()))
        xs = self.data_imgs[index]
        ys = self.data_masks[index]

        xs, ys = loader_npy.augment_flip(xs, ys)
        xs, ys = loader_npy.augment_brightness(xs, ys)
        xs, ys = loader_npy.elastic(xs, ys)
        
        # Add gray image channel, with shape [1, height, width]
        xs, ys = [item[np.newaxis, ...] for item in [xs, ys]]
        return torch.from_numpy(xs), torch.from_numpy(ys)

    def __getitem_val(self, index):
        return self.__getitem_train(index)

    def __getitem_infer(self, index):
        return

    def __getitem_test(self, index):
        return

    def __len__(self):
        return len(self.data_imgs)

    @staticmethod
    def elastic(xs, ys):
        img_rows, img_cols = xs.shape[0], xs.shape[1]
        x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')
        alpha = img_rows * 1.5
        sigma = img_rows * 0.07

        xs_shape = xs.shape
        blur_size = int(4 * sigma) | 1
        dx = cv2.GaussianBlur((np.random.rand(xs_shape[0], xs_shape[1]) * 2 - 1), ksize=(blur_size, blur_size),
                              sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(xs_shape[0], xs_shape[1]) * 2 - 1), ksize=(blur_size, blur_size),
                              sigmaX=sigma) * alpha
        if (x is None) or (y is None):
            x, y = np.meshgrid(np.arange(xs_shape[0]), np.arange(xs_shape[1]), indexing='ij')
        map_x = (x + dx).astype('float32')
        map_y = (y + dy).astype('float32')

        xs = cv2.remap(xs.astype('float64'), map_y, map_x, interpolation=cv2.INTER_NEAREST).reshape(xs_shape)
        ys = cv2.remap(ys.astype('float64'), map_y, map_x, interpolation=cv2.INTER_NEAREST).reshape(xs_shape)
        return xs, ys

    @staticmethod
    def augment_flip(xs, ys):
        xs_shape = xs.shape
        if len(xs_shape) == 2:
            xs, ys = [np.fliplr(item) for item in [xs, ys]]
        elif len(xs_shape) == 3:
            HWC_xs, HWC_ys = [np.transpose(item, [1, 2, 0]) for item in [xs, ys]]
            HWC_xs, HWC_ys = [np.fliplr(item) for item in [HWC_xs, HWC_ys]]
            xs, ys = [np.transpose(item, [2, 0, 1], ) for item in [HWC_xs, HWC_ys]]
        return xs, ys

    @staticmethod
    def augment_brightness(xs, ys):
        xs_shape = xs.shape
        if len(xs_shape) == 2:
            xbri = round(np.mean(xs) * 0.1)
            xbri = xbri.astype(int)
            xs1 = xs + xbri
            xs = np.where(xs1 > np.max(xs), np.max(xs), xs1)
        elif len(xs_shape) == 3:
            HWC_xs, HWC_ys = [np.transpose(item, [1, 2, 0]) for item in [xs, ys]]
            HWC_xs_shape = HWC_xs.shape
            xbri = round(np.mean(HWC_xs) * 0.1)
            xbri = xbri.astype(int)
            HWC_xs1 = HWC_xs + xbri
            for i in range(HWC_xs_shape[2]):
                HWC_xs[:, :, i] = np.where(HWC_xs1[:, :, i] > np.max(xs), np.max(xs), HWC_xs1[:, :, i])
            xs, ys = [np.transpose(item, [2, 0, 1], ) for item in [HWC_xs, HWC_ys]]
        return xs, ys
