# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "icdar17_train":{
            "data_dir":"MLT2019"
        },
        "icdar15_train":{
            "data_dir":"icdar2015"
        },
        "icdar15_test":{
            "data_dir":"icdar2015"
        },
        "art_test":{
            "data_dir":"ArT"
        },
        "icdar13_train":{
            "data_dir":"icdar2013"
        },
        "icdar13_test":{
            "data_dir":"icdar2013"
        },
        "totaltext_train":{
            "data_dir":"TotalText"
        },
        "totaltext_test":{
            "data_dir":"TotalText"
        },
        "synthtext800k":{
            "data_dir":"SynthText"
        },
        "synthtext150k":{
            "data_dir":"Synth150k"
        },
        "synthtext90k":{
            "data_dir":"SynthText_90KDict"
        },
        "synthtext90k_crop":{
            "data_dir":"SynthText_90KDict"
        },
        "synthtext_chinese":{
            "data_dir":"SynthText_Chinese"
        },
        "chinese_collect":{
            "data_dir":"chinese_collect"
        },
        "verisimilar":{
            "data_dir":"verisimilar"
        },
        "svt_train":{
            "data_dir":"SVT"
        },
        "svt_test":{
            "data_dir":"SVT"
        },
        "iiit_test":{
            "data_dir":"IIIT_STR_V1.0"
        },
        "cocotext_anno":{
            "data_dir":"cocotext_week_annotation_500"
        },
        "cocotext_test":{
            "data_dir":"cocotext_top500_retrieval"
        },
        "ctw_test":{
            "data_dir":"ctw_top100_retrieval"
        },
        "ctw_train":{
            "data_dir":"ctw_top100_retrieval"
        },
        "rctw_train":{
            "data_dir":"rctw"
        }
        # "cocotext_train":{
        #     "data_dir":"coco_text"
        # },
        # "cocotext_test":{
        #     "data_dir":"coco_text"
        # },
    }

    @staticmethod
    def get(name):
        if "icdar" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if "train" in name:
                is_train=True
            elif "test" in name:
                is_train=False
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=is_train,
                augment=None
            )
            if "15" in name:
                return dict(
                    factory="Icdar15Dateset",
                    args=args,
                )
            elif "17" in name:
                return dict(
                    factory="Icdar17Dateset",
                    args=args,
                )
            elif "13" in name:
                return dict(
                    factory="Icdar13Dateset",
                    args=args,
                )

        
        elif "rects" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="RectsDateset",
                args=args,
            )
        elif "TD500" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="TD500Dateset",
                args=args,
            )
        elif "ctw1500" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="Ctw1500Dateset",
                args=args,
            )
        elif "totaltext" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="TotalTextDateset",
                args=args,
            )
        elif "art" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="ArTDataset",
                args=args,
            )
        elif "mlt" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if "train" in name:
                is_train=True
            else:
                is_train=False
            args = dict(
                data_dir={os.path.join(data_dir, dataset) for dataset in attrs["data_dir"]},
                is_train=is_train
            )
            return dict(
                factory="MLTDateset",
                args=args,
            )
        # elif "synthtext" in name:
        #     data_dir = DatasetCatalog.DATA_DIR
        #     attrs = DatasetCatalog.DATASETS[name]
        #     args = dict(
        #         data_dir=os.path.join(data_dir, attrs["data_dir"]),
        #         is_train=True
        #     )
        #     return dict(
        #         factory="SynthTextDateset",
        #         args=args,
        #     )
        elif "cocotext_anno" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=False,
                augment=None
            )
            return dict(
                factory="COCOTextAnnoDateset",
                args=args,
            )
        elif "synthtext90k_crop" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=True,
                augment=None
            )
            return dict(
                factory="SynthText90kCropDateset",
                args=args,
            )
        elif "synthtext90k" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=True,
                augment=None
            )
            return dict(
                factory="SynthText90kDateset",
                args=args,
            )
        elif "synthtext800k" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=True,
                augment=None
            )
            return dict(
                factory="SynthTextDataset",
                args=args,
            )
        elif "synthtext150k" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=True,
                augment=None
            )
            return dict(
                factory="SynthText150kDataset",
                args=args,
            )
        elif "verisimilar" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=True,
                augment=None
            )
            return dict(
                factory="VeriSimilarDateset",
                args=args,
            )
        elif "synthtext_chinese" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=True,
                augment=None
            )
            return dict(
                factory="SynthtextChineseDataset",
                args=args,
            )
        elif "chinese_collect" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=False,
                augment=None
            )
            return dict(
                factory="ChineseCollectDataset",
                args=args,
            )
        elif "svt" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name),
                augment=None
            )
            return dict(
                factory="SVTDataset",
                args=args,
            )
        elif "iiit" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name),
                augment=None
            )
            return dict(
                factory="IIITDataset",
                args=args,
            )
        elif "cocotext" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="COCOTextDataset",
                args=args,
            )
        elif "rctw" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="RCTWDataset",
                args=args,
            )
        elif "ctw_train" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="CTWTrainDataset",
                args=args,
            )
        elif "ctw_test" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                is_train=("train" in name)
            )
            return dict(
                factory="CTWRetrievalDataset",
                args=args,
            )

        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
