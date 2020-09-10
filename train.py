#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
import os
import sys
import argparse

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
import config
import loader_npy
from models import build_model
sys.path.append("..")


conf = config.config()
conf.cuda = torch.cuda.is_available()
use_cuda = torch.cuda.is_available()
torch.manual_seed(conf.seed)


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        '--train_data_path',
        dest='train_data_path',
        default='./',
        type=str,
        help='path for training data'
    )
    parser.add_argument(
        '--train_label_path',
        dest='train_label_path',
        default='./',
        type=str,
        help='path for training labels'
    )
    parser.add_argument(
        '--val_data_path',
        dest='val_data_path',
        default='./',
        type=str,
        help='path for validation data'
    )
    parser.add_argument(
        '--val_label_path',
        dest='val_label_path',
        default='./',
        type=str,
        help='path for validation label'
    )

    return parser.parse_args()


def train(NetG, NetD, optimizerG, optimizerD, dataloader, epoch):
    total_dice = 0
    total_g_loss = 0
    total_g_loss_dice = 0
    total_g_loss_bce = 0
    total_d_loss = 0
    total_d_loss_penalty = 0
    NetG.train()
    NetD.train()

    for i, data in enumerate(dataloader, 1):
        # train D
        optimizerD.zero_grad()
        NetD.zero_grad()
        for p in NetG.parameters():
            p.requires_grad = False
        for p in NetD.parameters():
            p.requires_grad = True

        input, target = Variable(data[0]), Variable(data[1])
        input = input.float()
        target = target.float()

        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        output = NetG(input)
        output = F.sigmoid(output)
        output = output.detach()

        input_img = input.clone()
        output_masked = input_img * output
        if use_cuda:
            output_masked = output_masked.cuda()

        result = NetD(output_masked)

        target_masked = input_img * target
        if use_cuda:
            target_masked = target_masked.cuda()

        target_D = NetD(target_masked)
        loss_mac = - torch.mean(torch.abs(result - target_D))
        loss_mac.backward()

        # D net gradient_penalty
        batch_size = target_masked.size(0)
        gradient_penalty = utils.calc_gradient_penalty(NetD, target_masked, output_masked,
                                                       batch_size, use_cuda, input.shape)
        gradient_penalty.backward()
        optimizerD.step()

        # train G
        optimizerG.zero_grad()
        NetG.zero_grad()
        for p in NetG.parameters():
            p.requires_grad = True
        for p in NetD.parameters():
            p.requires_grad = False

        output = NetG(input)
        output = F.sigmoid(output)

        target_dice = target.view(-1).long()
        output_dice = output.view(-1)
        loss_dice = utils.dice_loss(output_dice, target_dice)

        output_masked = input_img * output
        if use_cuda:
            output_masked = output_masked.cuda()
        result = NetD(output_masked)

        target_G = NetD(target_masked)
        loss_G = torch.mean(torch.abs(result - target_G))
        loss_G_joint = loss_G + loss_dice
        loss_G_joint.backward()
        optimizerG.step()

        total_dice += 1 - loss_dice.data[0]
        total_g_loss += loss_G_joint.data[0]
        total_g_loss_dice += loss_dice.data[0]
        total_g_loss_bce += loss_G.data[0]
        total_d_loss += loss_mac.data[0]
        total_d_loss_penalty += gradient_penalty.data[0]

    for p in NetG.parameters():
        p.requires_grad = True
    for p in NetD.parameters():
        p.requires_grad = True

    size = len(dataloader)

    epoch_dice = total_dice / size
    epoch_g_loss = total_g_loss / size
    epoch_g_loss_dice = total_g_loss_dice / size
    epoch_g_loss_bce = total_g_loss_bce / size

    epoch_d_loss = total_d_loss / size
    epoch_d_loss_penalty = total_d_loss_penalty / size

    print_format = [epoch, conf.epochs, epoch_dice*100,
                    epoch_g_loss, epoch_g_loss_dice, epoch_g_loss_bce,
                    epoch_d_loss, epoch_d_loss_penalty]
    print('===> Training step {}/{} \tepoch_dice: {:.5f}'
          '\tepoch_g_loss: {:.5f} \tepoch_g_loss_dice: {:.5f}'
          '\tepoch_g_loss_bce: {:.5f} \tepoch_d_loss: {:.5f}'
          '\tepoch_d_loss_penalty: {:.5f}'.format(*print_format))


def val(NetG, dataloader_val, epoch, max_iou):
    IoUs, dices = [], []
    img_data = []
    NetG.eval()
    for i, data in enumerate(dataloader_val, 1):
        input, gt = Variable(data[0]), Variable(data[1])
        input = input.float()
        gt = gt.float()
        if use_cuda:
            input = input.cuda()
            gt = gt.cuda()
        pred = NetG(input)
        pred_np = np.where(pred.data > 0.5, 1, 0)

        gt = gt.data.cpu().numpy()
        cal_smooth = 1e-5
        for x in range(input.size()[0]):
            IoU = (np.sum(pred_np[x][gt[x] == 1]) + cal_smooth) / float(
                cal_smooth + np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x] == 1]))

            dice = (cal_smooth + np.sum(pred_np[x][gt[x] == 1]) * 2) / float(
                cal_smooth + np.sum(pred_np[x]) + np.sum(gt[x]))
            IoUs.append(IoU)
            dices.append(dice)
        outIMG = torch.from_numpy(np.where(pred.data > 0.5, 1, 0))

        image = data[0]
        label = data[1]
        img_data = [image, label, outIMG]

    IoUs = np.array(IoUs, dtype=np.float64)
    dices = np.array(dices, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    mdice = np.mean(dices, axis=0)
    print('mIoU: {:.4f}'.format(mIoU))
    print('Dice: {:.4f}'.format(mdice))
    if mIoU > max_iou:
        max_iou = mIoU
        if not os.path.exists(conf.outpath):
            os.makedirs(conf.outpath)
        torch.save(NetG.state_dict(), '%s/NetG_epoch_%d.pth' % (conf.outpath, epoch))

    # [dice, g_loss, d_loss, miou]
    result_data = np.array([mdice*100, None, None, None, None, None, mIoU * 100])
    print('===> ===> Validation Performance', '-' * 30,
          'mDice: %7.5f' % (mdice*100), '-' * 2,
          'mIoU: %7.5f' % (mIoU*100))
    return result_data, img_data, max_iou


def main(args):
    if use_cuda:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(conf.seed)
        cudnn.benchmark = True

    print('===> Building model')
    model = build_model(conf.model)
    NetG = model.NetD()
    NetD = model.NetG()
    print('===> Number of NetG params: {}'.format(
        sum([p.data.nelement() for p in NetG.parameters()])))
    print('===> Number of NetD params: {}'.format(
        sum([p.data.nelement() for p in NetD.parameters()])))

    if use_cuda:
        NetG = NetG.cuda()
        NetD = NetD.cuda()
    # setup optimizer
    decay = conf.decay
    optimizerG = optim.Adam(NetG.parameters(),
                            lr=conf.learning_rate, betas=(conf.beta1, 0.999))
    optimizerD = optim.Adam(NetD.parameters(),
                            lr=conf.learning_rate_netd, betas=(conf.beta1, 0.999))

    # load data
    training_set = DataLoader(
        dataset=loader_npy.loader_npy(image_path=args.train_data_path,
                                      mask_path=args.train_label_path,
                                      mode='train'),
        num_workers=conf.threads, batch_size=conf.batch_size,
        shuffle=True, pin_memory=True, drop_last=True)

    validation_set = DataLoader(
        dataset=loader_npy.loader_npy(image_path=args.val_data_path,
                                      mask_path=args.val_label_path,
                                      mode='val'),
        num_workers=conf.threads, batch_size=conf.batch_size,
        shuffle=True, pin_memory=True, drop_last=True)

    start_i = 1
    total_i = conf.epochs

    if conf.from_scratch:
        pass
    else:
        cp = utils.get_resume_path('s')
        pretrained_dict = torch.load(cp)
        model_dict = NetG.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        NetG.load_state_dict(model_dict)

        cp_name = os.path.basename(cp)

        cp2 = utils.get_resume_path('c')
        print('#####  resume_path(c):', cp2)
        NetD.load_state_dict(torch.load(cp2))
        print('---> Loading checkpoint {}...'.format(cp_name))
        start_i = int(cp_name.split('_')[-1].split('.')[0]) + 1

    max_iou = 0
    print('===> Begin training at epoch {}'.format(start_i))
    for epoch in range(start_i, total_i + 1):
        print("---------------eppch[{}]-------------------".format(epoch))
        train(NetG, NetD, optimizerG, optimizerD, training_set, epoch)

        if epoch % 2 == 0:
            val(NetG, validation_set, epoch, max_iou)

            utils.save_checkpoints(NetG, epoch, 's')
            utils.save_checkpoints(NetD, epoch, 'c')

        if epoch % 20 == 0:
            conf.learning_rate = conf.learning_rate * decay
            if conf.learning_rate <= 0.00000001:
                conf.learning_rate = 0.00000001

            conf.learning_rate_netd = conf.learning_rate_netd * decay
            if conf.learning_rate_netd <= 0.00000001:
                conf.learning_rate_netd = 0.00000001

            print('Learning Rate: {:.6f}'.format(conf.learning_rate))
            optimizerG = optim.Adam(NetG.parameters(), lr=conf.learning_rate, betas=(conf.beta1, 0.999))
            optimizerD = optim.Adam(NetD.parameters(), lr=conf.learning_rate_netd, betas=(conf.beta1, 0.999))


if __name__ == '__main__':
    main(get_args())
