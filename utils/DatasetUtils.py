"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def get_params(cfg, size):
    w, h = size
    new_h = h
    new_w = w
    preprocess = cfg['preprocess']
    if preprocess == 'resize_and_crop':
        new_h = new_w = cfg['load_size']
    elif preprocess == 'scale_width_and_crop':
        new_w = cfg['load_size']
        new_h = new_w * h // w

    x = random.randint(0, np.maximum(0, new_w - cfg['crop_size']))
    y = random.randint(0, np.maximum(0, new_h - cfg['crop_size']))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(cfg, params=None, grayscale=False, method=InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    preprocess = cfg['preprocess']
    if 'resize' in preprocess:
        osize = [cfg['load_size'], cfg['load_size']]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, cfg['load_size'], cfg['crop_size'], method)))

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(cfg['crop_size']))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], cfg['crop_size'])))

    if preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if cfg['flip']:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True