#!/usr/bin/env python
# encoding: utf-8
import os

import cv2
import torch
import scipy
import dicom
import numpy as np
import SimpleITK as sitk
from skimage import morphology
from PIL import Image


import config
from functools import partial


# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(y_conv, y_true):
    y_true = y_true.float()
    intersection = torch.sum(y_conv * y_true)

    smooth = 1e-5
    union = torch.sum(y_conv * y_conv) + torch.sum(y_true * y_true)
    # dice = 2.0 * intersection / union
    dice = 2 * (smooth + intersection) / (union + smooth)

    return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7)


def get_resume_path(c_type):
    """Return latest checkpoints by default otherwise return the specified one."""
    conf = config.config()
    names = [os.path.join(conf.checkpoint_dir, p) for p in os.listdir(conf.checkpoint_dir)]
    require = os.path.join(conf.checkpoint_dir, conf.prefix + c_type + '_' + str(conf.resume_step) + '.pth')
    if conf.resume_step == -1:
        list_name = sorted(names, key=os.path.getmtime)
        for i in reversed(range(len(list_name))):
            if list_name[i].find(conf.prefix + c_type) >= 0:
                return list_name[i]

    elif os.path.isfile(require):
        return require
    raise Exception('\'%s\' dose not exist!' % require)


def save_checkpoints(model, step, type):
    conf = config.config()
    if not os.path.exists(conf.checkpoint_dir):
        os.makedirs(conf.checkpoint_dir)

    # Recommand: save and load only the model parameters
    filename = conf.prefix + type + '_' + str(step) + '.pth'
    torch.save(model.state_dict(), os.path.join(conf.checkpoint_dir, filename))
    print("===> ===> ===> Save checkpoint {} to {}".format(step, filename))


def showTrain(vis, train_img, train_results, epoch, wins_train_im, wins):
    basic_opts = partial(dict, xlabel='epoch', legend=['train'])

    train_stats = ['dice', 'G_Loss', 'D_Loss']
    # Windows for images
    im_titles = ['input', 'label', 'prediction']
    if epoch == 1:
        wins_train_im = [vis.images(item, opts=dict(title='train_' + im_titles[j])) for j, item in
                         enumerate(train_img)]
        for j, stat in enumerate(train_stats):
            wins.append(vis.line(X=np.array([epoch]), Y=train_results[None, j], opts=basic_opts(title=stat)))
    else:
        if len(wins_train_im) == 0:
            wins_train_im = [vis.images(item, opts=dict(title='train_' + im_titles[j])) for j, item in
                             enumerate(train_img)]
        else:
            for j, item in enumerate(train_img):
                vis.images(item, opts=dict(title='train_' + im_titles[j]), win=wins_train_im[j])

        for j, win in enumerate(wins):
            vis.updateTrace(X=np.array([epoch]), Y=train_results[None, j], win=win, name='train')

    return wins_train_im, wins


def showVal(vis, val_img, val_results, epoch, wins_val_im, wins):
    basic_opts = partial(dict, xlabel='epoch', legend=['val'])
    im_titles = ['input', 'label', 'prediction']
    val_stat = ['dice', 'Mou']
    if epoch == 1:
        wins_val_im = [vis.images(item, opts=dict(title='val_' + im_titles[j]))
                       for j, item in enumerate(val_img)]
        for j, stat in enumerate(val_stat):
            wins.append(vis.line(X=np.array([epoch]), Y=val_results[None, j], opts=basic_opts(title=val_stat)))
    else:
        for j, win in enumerate(wins):
            vis.updateTrace(X=np.array([epoch]), Y=val_results[None, j], win=win, name='val')
        if len(wins_val_im) == 0:

            wins_val_im = [vis.images(item, opts=dict(title='val_' + im_titles[j]))
                           for j, item in enumerate(val_img)]
        else:
            for j, item in enumerate(val_img):
                vis.images(item, opts=dict(title='val_' + im_titles[j]), win=wins_val_im[j])

    return wins_val_im, wins


def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, use_cuda, input_shape, LAMBDA=10):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement() // BATCH_SIZE)
    alpha = alpha.contiguous().view(input_shape)

    alpha = alpha.cuda() if use_cuda else alpha

    if isinstance(real_data, torch.autograd.Variable):
        real_data = real_data.data

    if isinstance(fake_data, torch.autograd.Variable):
        fake_data = fake_data.data

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()

    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(disc_interpolates.size()),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def copy_normalized(src, dtype=np.int16):
    src_shape = np.shape(src)
    if src_shape == shape_max:
        return src

    (z_axis, y_axis, x_axis) = src_shape
    new_img = np.full(shape_max, np.min(src), dtype=dtype)
    if z_axis < Z_MAX:
        start = int((Z_MAX - z_axis) / 2)
        for i in range(z_axis):
            copy_slice_centered_custom(new_img[start + i], src[i], x_axis, y_axis)
    else:
        start = int((z_axis - Z_MAX) / 2)
        for i in range(Z_MAX):
            copy_slice_centered_custom(new_img[i], src[start + i], x_axis, y_axis)
    return new_img


def copy_slice_centered_custom(dst, src, dim_x, dim_y):
    if dim_x <= X_MAX and dim_y <= Y_MAX:
        x_start = int((X_MAX - dim_x) / 2)
        y_start = int((Y_MAX - dim_y) / 2)
        for y in range(dim_y):
            for x in range(dim_x):
                dst[y_start+y][x_start+x] = src[y][x]
    elif dim_x > X_MAX and dim_y > Y_MAX:
        x_start = int((dim_x - X_MAX) / 2)
        y_start = int((dim_y - Y_MAX) / 2)
        for y in range(Y_MAX):
            for x in range(X_MAX):
                dst[y][x] = src[y_start+y][x_start+x]
    else:
        if dim_x <= X_MAX and dim_y > Y_MAX:
            x_start = int((X_MAX - dim_x) / 2)
            y_start = int((dim_y - Y_MAX) / 2)
            for y in range(Y_MAX):
                for x in range(dim_x):
                    dst[y][x_start+x] = src[y_start+y][x]
        else:
            x_start = int((dim_x - X_MAX) / 2)
            y_start = int((Y_MAX - dim_y) / 2)
            for y in range(dim_y):
                for x in range(X_MAX):
                    dst[y_start+y][x] = src[y][x_start+x]


def slice_normalize(slice, bounds=None):
    b, t = np.percentile(slice, (0.0, 100.0))
    if bounds != None:
        min_bound, max_bound = bounds
        if min_bound != None:
            b = min_bound
        if max_bound != None:
            t = max_bound

    slice = np.clip(slice, b, t)
    if np.std(slice) == 0:
        return slice
    else:
        return (slice - np.mean(slice)) / np.std(slice)


def resample_volume_custom(img, new_shape, spacing_old, spacing_new, bounds=None, normalize=True):

    global Z_MAX, Y_MAX, X_MAX, shape_max
    Z_MAX = int(new_shape[0])
    Y_MAX = int(new_shape[1])
    X_MAX = int(new_shape[2])
    shape_max = (Z_MAX, Y_MAX, X_MAX)
    resize_factor = np.array(spacing_old) / spacing_new
    new_shape_custom = np.round(np.shape(img) * resize_factor)
    real_resize_factor = new_shape_custom / np.shape(img)
    img_rescaled = scipy.ndimage.interpolation.zoom(img, real_resize_factor, order=0, mode='nearest')
    img_array_normalized = copy_normalized(img_rescaled)

    img_tmp = img_array_normalized.copy()

    if normalize:
        img_tmp = slice_normalize(img_tmp, bounds)
    return img_tmp


def after_effect(new_img, min_size=250):
    black_white = []
    for i in range(new_img.shape[0]):
        black_white.append(np.sum(new_img[i, :, :]))
    index_TF = np.where(np.array(black_white) > 0)[0]
    for index in index_TF:
        index = index-1
        pixel_num = np.sum(new_img[index, :, :])
        if pixel_num > min_size * 2:
            new_img[index, :, :] = morphology.remove_small_objects(new_img[index, :, :] >= 1, min_size=min_size, connectivity=1)
        last_image = np.where(new_img[index, :, :] >= 1, 255, 0)
        last_image = Image.fromarray(last_image.astype('uint8')).convert('RGB')
        last_image = np.asarray(last_image)
        last_image = cv2.medianBlur(last_image, 5)
        last_image = np.where(last_image[:, :, 1] >= 1, 1, 0)
        new_img[index, :, :] = (last_image > 0.5).astype(np.uint8)
    return new_img


def copy_dicommetadata(orgimg, tarimage):
    orgimg.SetMetaData('0008|0016', tarimage.GetMetaData('0008|0016'))
    orgimg.SetMetaData('0008|0018', tarimage.GetMetaData('0008|0018'))

    orgimg.SetMetaData('0020|0013', tarimage.GetMetaData('0020|0013'))
    orgimg.SetMetaData('0020|0012', tarimage.GetMetaData('0020|0012'))
    orgimg.SetMetaData('0020|0011', tarimage.GetMetaData('0020|0011'))
    orgimg.SetMetaData('0008|0008', tarimage.GetMetaData('0008|0008'))

    orgimg.SetMetaData('0020|000d', tarimage.GetMetaData('0020|000d'))
    orgimg.SetMetaData('0020|000e', tarimage.GetMetaData('0020|000e'))

    orgimg.SetMetaData('0018|0050', tarimage.GetMetaData('0018|0050'))
    orgimg.SetMetaData('0018|0088', tarimage.GetMetaData('0018|0088'))
    orgimg.SetMetaData('0028|0030', tarimage.GetMetaData('0028|0030'))
    orgimg.SetMetaData('0020|1040', tarimage.GetMetaData('0020|1040'))
    orgimg.SetMetaData('0020|1041', tarimage.GetMetaData('0020|1041'))
    orgimg.SetMetaData('0020|0032', tarimage.GetMetaData('0020|0032'))
    orgimg.SetMetaData('0020|0037', tarimage.GetMetaData('0020|0037'))
    return orgimg


def save_updated_image_custom(img_arr, path, origin, spacing, dicom_image=None, pixel_type=sitk.sitkFloat64,
                              rescale_intensity=True):
    itk_scaled_img = sitk.GetImageFromArray(img_arr, isVector=False)
    itk_scaled_img.SetSpacing(spacing)
    itk_scaled_img.SetOrigin(origin)
    if (dicom_image != None):
        itk_scaled_img.SetDirection(dicom_image.GetDirection())
        itk_scaled_img = copy_dicommetadata(itk_scaled_img, dicom_image)

        castFilter = sitk.CastImageFilter()
        castFilter.SetOutputPixelType(sitk.sitkUInt16)
        itk_scaled_img = castFilter.Execute(itk_scaled_img)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if rescale_intensity:
        itk_scaled_img = sitk.RescaleIntensity(itk_scaled_img)

    itk_scaled_img = sitk.Cast(itk_scaled_img, pixel_type)
    sitk.WriteImage(itk_scaled_img, path)


def save_predict_data(predict_img, save_path, predict_spacing
                      , org_shape, org_spacing, org_origin
                      , pix_type=0):
    new_img_spacing = (org_spacing[2], org_spacing[1], org_spacing[0])
    old_img_spacing = predict_spacing

    new_img = resample_volume_custom(predict_img,
                                     np.array([org_shape[0], org_shape[1], org_shape[2]], dtype=np.float32),
                                     old_img_spacing, new_img_spacing, normalize=False)

    pix_type_sitk = sitk.sitkInt16
    if pix_type == 1:
        pix_type_sitk = sitk.sitkInt32
    elif pix_type == 2:
        pix_type_sitk = sitk.sitkFloat64

    new_img = (new_img > 0.5).astype(np.uint8)

    new_img = after_effect(new_img)
    new_img = np.where(new_img > 0, 2, 0)
    save_updated_image_custom(img_arr=new_img, path=save_path, origin=org_origin, spacing=org_spacing,
                              pixel_type=pix_type_sitk, rescale_intensity=False)


def get_name_from_orgpath(file_temp):
    file_name = os.path.basename(file_temp)
    if not file_name.endswith('.nii.gz'):
        file_name = file_name + '_prostate.nii.gz'
    return file_name


def dicom_get_pixels(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    return image


def normalize_method(xs):
    if xs.dtype != np.float64:
        xs = xs.astype(np.float64)

    if np.max(xs) - np.min(xs) != 0.0:
        xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))
    else:
        if np.min(xs) < 0:
            xs = xs * 0
        elif np.max(xs) > 1:
            xs = xs / np.max(xs)

    return xs


def get_normlized_data(org_path, Y_MAX, X_MAX, vox_spacing,
                       MIN_BOUND=None, MAX_BOUND=None, normalize=True):
    org_img = []
    org_img_spacing = ()
    org_img_origin = ()

    if os.path.isdir(org_path):  # dicom

        slices, attr = dicom_load_scan(org_path)

        org_img = dicom_get_pixels(slices)
        org_img_spacing = attr['Spacing']
        org_img_origin = attr['Origin']

    elif org_path.endswith('.mhd'):

        itk_img = sitk.ReadImage(org_path)

        org_img = sitk.GetArrayFromImage(itk_img)
        org_img_spacing = itk_img.GetSpacing()
        org_img_origin = itk_img.GetOrigin()

    org_img_shape = org_img.shape
    new_img_spacing = (org_img_spacing[2], vox_spacing, vox_spacing)
    old_img_spacing = (org_img_spacing[2], org_img_spacing[1], org_img_spacing[0])

    new_img = resample_volume_custom(org_img, np.array([org_img_shape[0], Y_MAX, X_MAX], dtype=np.float32),
                                     old_img_spacing,
                                     new_img_spacing, bounds=(MIN_BOUND, MAX_BOUND), normalize=normalize)
    # normlize
    new_img = normalize_method(new_img)

    return new_img, new_img_spacing, org_img_shape, org_img_spacing, org_img_origin


def dicom_load_scan(path):
    attr = {}
    slices = []
    for s in os.listdir(path):
        if s.endswith('.DS_Store'):
            continue
        slices.append(dicom.read_file(path + '/' + s))
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    slices2 = []
    prev = -1000000
    # remove redundant slices
    for slice in slices:
        cur = slice.ImagePositionPatient[2]
        if cur == prev:
            continue
        prev = cur
        slices2.append(slice)
    slices = slices2

    for i in range(len(slices)-1):
        try:
            slice_thickness = np.abs(slices[i].ImagePositionPatient[2] - slices[i+1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[i].SliceLocation - slices[i+1].SliceLocation)
        if slice_thickness != 0:
            break

    assert slice_thickness != 0

    for s in slices:
        s.SliceThickness = slice_thickness

    x, y = slices[0].PixelSpacing
    attr['Spacing'] = (x, y, slice_thickness)
    attr['Origin'] = slices[0].ImagePositionPatient

    return slices, attr
