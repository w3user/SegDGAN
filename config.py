#!/usr/bin/env python
# encoding: utf-8
# Training configuration
import torch


class config:
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.model = 'segdensean'
        self.batch_size = 8
        self.epochs = 800
        self.augment_size = 500
        self.learning_rate = 3e-3
        self.learning_rate_netd = 3e-3
        self.decay = 0.5
        self.criterion = 'dice'
        self.beta1 = 0.5
        self.seed = 714
        self.threads = 0
        self.from_scratch = False
        self.checkpoint_dir = './checkpoints/'
        self.result_dir = './results/'
        self.outpath = './outpath/'
        self.out_val_path = './outpath/val/'
        self.out_train_path = './outpath/train/'

        self.resume_step = -1
        self.prefix = 'GAN'
