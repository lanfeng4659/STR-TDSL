# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference_two_net import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.config import cfg as cfg_det
# from maskrcnn_benchmark.config import cfg as cfg_ret
# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

def construct_network(cfg,cfg_file):
    
    # print(dir(cfg))
    # exit()
    # cfg.clear()
    # args.config_file = cfg_file #'configs/detect.yaml'
    # print(cfg.MODEL.RETRIEVAL_ONLY)
    cfg.setdefault(cfg_file)
    # print(cfg.MODEL.RETRIEVAL_ONLY)
    cfg.merge_from_file(cfg_file)
    # print(cfg.MODEL.RETRIEVAL_ONLY)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = None
    # print(ckpt)
    _ = checkpointer.load(ckpt, use_latest=True)
    return model,cfg

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    # parser.add_argument(
    #     "--config-file",
    #     default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
    #     metavar="FILE",
    #     help="path to config file",
    # )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )	# --config-file "configs/align/line_bezier0732.yaml" 
	# --skip-test \
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    # print(dir(cfg_det))
    cfg_ret = cfg_det.clone()
    retrieval_model,cfg = construct_network(cfg_ret,'configs/retrieval_only.yaml')
    detect_model,cfg = construct_network(cfg_det,'configs/detect.yaml')
    # logger.info(cfg_det)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)


    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON and not cfg.MODEL.KE_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.KE_ON:
        iou_types = iou_types + ("kes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    rec_type = cfg.MODEL.ALIGN.PREDICTOR
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        cur_result = inference(
            detect_model,
            retrieval_model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            rec_type = rec_type,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
        print("current mAP:{}\n".format(cur_result))


if __name__ == "__main__":
    main()
