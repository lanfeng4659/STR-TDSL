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
# def crop_image
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.poolers import Pooler
import numpy as np
def crop_image(images,boxes):
    image = images.tensors
    bboxes = boxes.bbox.data.cpu().numpy().astype(np.int32)
    
    all_images = []
    for box in bboxes:
        minx,miny,maxx,maxy = box
        all_images.append(torch.nn.functional.interpolate(image[:,:,miny:maxy,minx:maxx], size=(4*32,15*32),mode='bilinear', align_corners=True)) 
    return to_image_list(torch.cat(all_images,dim=0))
def compute_on_dataset(model_detect, model_retrieval, data_loader, device, timer=None):
    model_detect.eval()
    model_retrieval.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        
        with torch.no_grad():
            if timer:
                timer.tic()
            # output = model(images.to(device),targets,is_words=(_==0))
            output = model_detect(images.to(device),targets,is_words=True)
            if output[0].bbox.size(0)==0:
                zero_bedding = torch.zeros([0,1920]).type_as(output[0].bbox)
                output[0].add_field("imgs_embedding_nor",zero_bedding)
                # output[0].add_field("words_embedding_nor",retrieval_result["words_embedding_nor"])
            else:
                all_images = crop_image(images,output[0])
                retrieval_result = model_retrieval(all_images.to(device),targets,is_words=True)
                output[0].add_field("imgs_embedding_nor",retrieval_result["imgs_embedding_nor"])
                output[0].add_field("words_embedding_nor",retrieval_result["words_embedding_nor"])
            # print(output,images)
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
        model_detect,
        model_retrieval,
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
    predictions = compute_on_dataset(model_detect, model_retrieval, data_loader, device, inference_timer)
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
