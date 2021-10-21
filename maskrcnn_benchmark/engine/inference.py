# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.structures.image_list import to_image_list
def resize_image(images,new_size, targets):
    image_tensors = images.tensors
    new_images = torch.nn.functional.interpolate(image_tensors, size=(new_size[0],new_size[1]), mode='bilinear', align_corners=True)
    return to_image_list(new_images)
def multi_scale_compute_on_dataset(model, data_loader, device, timer=None, scales=(1280,960, 1600)):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        
        with torch.no_grad():
            if timer:
                timer.tic()
            # output = model(images.to(device),targets,is_words=(_==0))
            # print(images.tensors.shape)
            b,c,h,w = images.tensors.shape
            longer_side = max([h,w])

            output = model(images.to(device),targets,is_words=(_==0))
            for side in scales:
                if side != longer_side:
                    if h > w:
                        new_h = side
                        new_w = int(side/longer_side*w)
                    else:
                        new_w = side
                        new_h = int(side/longer_side*h)
                    pad_h = new_h if new_h%32==0 else (new_h//32+1)*32
                    pad_w = new_w if new_w%32==0 else (new_w//32+1)*32
                    images = resize_image(images,[pad_h,pad_w], targets)
                    output_scale = model(images.to(device),targets,is_words=False)
                    output[0].add_field("imgs_embedding_nor",torch.cat([output[0].get_field("imgs_embedding_nor"),output_scale[0].get_field("imgs_embedding_nor")]))
            # import ipdb; ipdb.set_trace()
                    
            if _ == 0:
                output[0].add_field("y_trues",targets[0].get_field("y_trues"))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    
    return results_dict

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        
        with torch.no_grad():
            if timer:
                timer.tic()
            # output = model(images.to(device),targets,is_words=(_==0))
            # print(images.tensors.shape)
            output = model(images.to(device),targets,is_words=True)
            if _ == 0:
                output[0].add_field("y_trues",targets[0].get_field("y_trues"))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        rec_type = "ctc",
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        rec_type=rec_type,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    # load predictions if exists
    prediction_file = os.path.join(output_folder, 'predictions.pth')
    # if os.path.isfile(prediction_file):
    #     predictions = torch.load(prediction_file)
    #     logger.info("Found prediction results at {}".format(prediction_file))

    #     return evaluate(dataset=dataset,
    #                     predictions=predictions,
    #                     output_folder=output_folder,
    #                     **extra_args)

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    # predictions = multi_scale_compute_on_dataset(model, data_loader, device, inference_timer)
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
