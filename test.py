#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from skimage.transform import resize

sys.path.append("..")
from models import build_model
from utils import save_predict_data
from utils import get_normlized_data
from utils import get_name_from_orgpath

use_cuda = torch.cuda.is_available()


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        '--test_data_path',
        dest='test_data_path',
        default='./',
        type=str,
        help='path to test data'
    )
    parser.add_argument(
        '--resume_node',
        dest='resume_node',
        default='./',
        type=str,
        help='path to model checkpoint'
    )
    parser.add_argument(
        '--model',
        dest='model',
        default='./',
        type=str,
        help='model name'
    )
    parser.add_argument(
        '--save_path',
        dest='save_path',
        default='./',
        type=str,
        help='path for save predicted data'
    )

    return parser.parse_args()


def get_result_from_model(model, new_img_for_pred, model_output_number=2, model_output_index=0):
    new_img_for_pred = torch.from_numpy(new_img_for_pred)

    new_img_for_pred = Variable(new_img_for_pred, volatile=True).float()
    if use_cuda:
        new_img_for_pred = new_img_for_pred.cuda()

    if model_output_number > 1:
        *pre_results, = model(new_img_for_pred)
        if model_output_index >= 0:
            result_image_array = pre_results[model_output_index]
        else:
            result_image_array = pre_results[len(pre_results) - 1]
    else:
        result_image_array = model(new_img_for_pred)
    return result_image_array


def model_predict_method(model, input_path, save_path, Y_MAX, X_MAX,
                         vox_spacing, MIN_BOUND=None, MAX_BOUND=None,
                         normalize=False, model_type=3, model_output_number=1,
                         model_output_index=-1, model_output_flattened=True,
                         model_output_dim=2, Z_resize=None, Y_resize=None,
                         X_resize=None, batchsize_2D=None):
    model.eval()
    for file_temp in os.listdir(input_path):
        if file_temp.endswith('.DS_Store'):
            continue
        elif os.path.isdir(os.path.join(input_path, file_temp)) or file_temp.endswith('.mhd'):
            print('############### file_temp', file_temp)
            file_temp = os.path.join(input_path, file_temp)

            new_img, new_img_spacing, org_img_shape, org_img_spacing, org_img_origin = get_normlized_data(
                org_path=file_temp, Y_MAX=Y_MAX, X_MAX=X_MAX
                , vox_spacing=vox_spacing
                , MIN_BOUND=MIN_BOUND, MAX_BOUND=MAX_BOUND
                , normalize=normalize)
            new_img_shape = new_img.shape
            input_resize = False
            org_old_shape = new_img.shape
            if Z_resize == None and Y_resize == None and X_resize == None:
                pass
            else:
                input_resize = True
                if Z_resize == None:
                    Z_resize = new_img_shape[0]
                if Y_resize == None:
                    Y_resize = Y_MAX
                if X_resize == None:
                    X_resize = X_MAX
                new_img = resize(new_img, (Z_resize, Y_resize, X_resize), mode='constant')

            result_image_array = []

            if model_type == 3:
                new_img_for_pred = new_img[np.newaxis, np.newaxis, ...]
                result_image_array = get_result_from_model(model, new_img_for_pred,
                                                           model_output_number, model_output_index)

            elif model_type == 2:

                new_img_for_pred = new_img[np.newaxis, ...]
                new_img_for_pred = np.transpose(new_img_for_pred, [1, 0, 2, 3])

                # batch_size
                data_size = new_img_for_pred.shape[0]
                if batchsize_2D == None:
                    batchsize_2D = new_img_shape[0]
                if data_size > batchsize_2D:
                    loop = int(data_size / batchsize_2D)
                    for i in range(0, loop):
                        new_img_for_pred_T = new_img_for_pred[i * batchsize_2D:(i + 1) * batchsize_2D, :, :, :]
                        result_image_array_T = get_result_from_model(model, new_img_for_pred_T, model_output_number,
                                                                     model_output_index)

                        if isinstance(result_image_array_T, torch.autograd.Variable):
                            result_image_array_T = result_image_array_T.data
                        result_image_array.extend(result_image_array_T.cpu().numpy())

                    if loop * batchsize_2D < data_size:
                        new_img_for_pred_T = new_img_for_pred[loop * batchsize_2D:data_size, :, :, :]
                        result_image_array_T = get_result_from_model(model, new_img_for_pred_T, model_output_number,
                                                                     model_output_index)
                        if isinstance(result_image_array_T, torch.autograd.Variable):
                            result_image_array_T = result_image_array_T.data
                        result_image_array.extend(result_image_array_T.cpu().numpy())

                    result_image_array = torch.from_numpy(np.array(result_image_array))

                elif data_size <= batchsize_2D:
                    result_image_array = get_result_from_model(model, new_img_for_pred, model_output_number,
                                                               model_output_index)

            if model_output_flattened == False:
                result_image_array = result_image_array.squeeze()
                result_image_array = result_image_array.view(-1, model_output_dim)
            if model_output_dim == 1:
                if isinstance(result_image_array, torch.autograd.Variable):
                    result_image_array = result_image_array.data.cpu()
                result_image_array = torch.from_numpy(
                    np.where(result_image_array.numpy() > 0.5, 1.0, 0.0).astype(np.float64)
                )
            elif model_output_dim > 1:
                if isinstance(result_image_array, torch.autograd.Variable):
                    result_image_array = result_image_array.data
                result_image_array = result_image_array.max(1)[1]

            result_image_array = result_image_array.cpu().view(*new_img.shape)

            if isinstance(result_image_array, torch.autograd.Variable):
                result_image_array = result_image_array.data
            result_image_array = result_image_array.numpy()
            if input_resize:
                result_image_array = resize(result_image_array, org_old_shape, mode='constant')

            save_predict_data(predict_img=result_image_array,
                              save_path=os.path.join(save_path, get_name_from_orgpath(file_temp)),
                              predict_spacing=new_img_spacing,
                              org_shape=org_img_shape,
                              org_spacing=org_img_spacing,
                              org_origin=org_img_origin
                              )


def pred(args):

    Y_MAX = 128
    X_MAX = 128
    vox_spacing = 1
    MIN_BOUND = None
    MAX_BOUND = None
    normalize = True
    model_type = 2
    model_output_number = 1
    model_output_index = 0
    model_output_flattened = False
    model_output_dim = 1
    batchsize_2D = 1

    model = build_model(args.model)
    model = model.NetS()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        model = model.cuda()

    pretrained_dict = torch.load(args.resume_node)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model_predict_method(model=model,
                         input_path=args.test_data_path,
                         save_path=args.save_path,
                         Y_MAX=Y_MAX,
                         X_MAX=X_MAX,
                         vox_spacing=vox_spacing,
                         MIN_BOUND=MIN_BOUND,
                         MAX_BOUND=MAX_BOUND,
                         normalize=normalize,
                         model_type=model_type,
                         model_output_number=model_output_number,
                         model_output_index=model_output_index,
                         model_output_dim=model_output_dim,
                         model_output_flattened=model_output_flattened,
                         batchsize_2D=batchsize_2D)


if __name__ == '__main__':
    pred(get_args())
