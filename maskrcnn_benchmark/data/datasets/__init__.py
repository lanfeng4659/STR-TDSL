# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .concat_dataset import ConcatDataset
from .synthtext90k import SynthText90kDateset
from .svt import SVTDataset
from .iiit import IIITDataset
from .cocotext import COCOTextDataset
from .synthtext_chinese import SynthtextChineseDataset
from .chinese_collect import ChineseCollectDataset
from .ic15 import Icdar15Dateset
from .art import ArTDataset
__all__ = ["ConcatDataset","SynthtextChineseDataset", 
"SynthText90kDateset","SVTDataset","IIITDataset","COCOTextDataset", "Icdar15Dateset", "ArTDataset"]
