# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


# def build_transforms(cfg, is_train=True):
#     if is_train:
#         if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
#             min_size = cfg.INPUT.MIN_SIZE_TRAIN
#         else:
#             assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
#                 "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
#             min_size = range(
#                 cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
#                 cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
#             )
#         max_size = cfg.INPUT.MAX_SIZE_TRAIN

#         flip_horizontal_prob = cfg.INPUT.FLIP_PROB_TRAIN
#         flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
#         crop_prob = cfg.INPUT.CROP_PROB_TRAIN
#         brightness = cfg.INPUT.BRIGHTNESS
#         contrast = cfg.INPUT.CONTRAST
#         saturation = cfg.INPUT.SATURATION
#         hue = cfg.INPUT.HUE
#     else:
#         min_size = cfg.INPUT.MIN_SIZE_TEST
#         max_size = cfg.INPUT.MAX_SIZE_TEST
#         flip_horizontal_prob = 0.0
#         flip_vertical_prob = 0.0
#         crop_prob = 0
#         brightness = 0.0
#         contrast = 0.0
#         saturation = 0.0
#         hue = 0.0

#     to_bgr255 = cfg.INPUT.TO_BGR255
#     normalize_transform = T.Normalize(
#         mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
#     )
#     color_jitter = T.ColorJitter(
#         brightness=brightness,
#         contrast=contrast,
#         saturation=saturation,
#         hue=hue,
#     )
#     if is_train:
#         transform = T.Compose(
#             [
#                 # color_jitter,
#                 # T.Resize(min_size, max_size),
#                 # T.RandomCropExpand(crop_prob),
#                 # T.RandomHorizontalFlip(flip_horizontal_prob),
#                 # T.RandomVerticalFlip(flip_vertical_prob),
#                 T.ToTensor(),
#                 normalize_transform,
#             ]
#         )
#     else:
#         transform = T.Compose(
#             [
#                 T.ToTensor(),
#                 normalize_transform,
#             ]
#         )
#     return transform

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        rotate_prob = 0.2

        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    
    if is_train:
        color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        out_s, min_s, max_s = 640, 210, 1100
        resize_method=T.RandomCrop_resize(out_s,min_s,max_s)
        transform = T.Compose(
            [
                color_jitter,
                # resize_method,
                # T.RandomRotate(rotate_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform