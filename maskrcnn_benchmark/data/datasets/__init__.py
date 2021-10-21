# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .concat_dataset import ConcatDataset
from .word_dataset import WordDataset
from .ic15 import Icdar15Dateset
from .ic13 import Icdar13Dateset
from .ic17 import Icdar17Dateset
from .total_text import TotalTextDateset
from .synthtext90k import SynthText90kDateset
from .verisimilar import VeriSimilarDateset
from .svt import SVTDataset
from .iiit import IIITDataset
from .cocotext import COCOTextDataset
from .rctw import RCTWDataset
from .ctw_train import CTWTrainDataset
from .ctw_test import CTWRetrievalDataset
from .synthtext_chinese import SynthtextChineseDataset
from .chinese_collect import ChineseCollectDataset
from .synthtext90k_crop import SynthText90kCropDateset
from .coco_annotation import COCOTextAnnoDateset
from .synthtext800k import SynthTextDataset
from .synth150k import SynthText150kDataset
__all__ = ["ConcatDataset", "WordDataset","TotalTextDateset","Icdar13Dateset","Icdar15Dateset","Icdar17Dateset","SynthtextChineseDataset", "SynthText150kDataset",
"SynthText90kDateset","SynthTextDataset","VeriSimilarDateset","SVTDataset","IIITDataset","COCOTextDataset","RCTWDataset","CTWTrainDataset","CTWRetrievalDataset",
"ChineseCollectDataset",'SynthText90kCropDateset','COCOTextAnnoDateset']
