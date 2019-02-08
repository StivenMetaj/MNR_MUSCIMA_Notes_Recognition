# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]

            ''' PROVIAMO A CAPIRE MEGLIO LE PREDIZIONI
            '''
            # import convert_to_coco
            # convert_to_coco.clearDirectory("../data/mnr/PROVESTIV")
            # for j, ou in enumerate(output):
            #     imgID = data_loader.dataset.id_to_img_map[i*len(batch) + j]
            #     imgPath = "../data/mnr/test2019/" + '{0:012d}'.format(imgID) + ".jpg"
            #     import matplotlib.pyplot as plt
            #     import matplotlib.image as mpimg
            #     img = mpimg.imread(imgPath)
            #     plt.imshow(img, cmap="gray")
            #     plt.savefig("../data/mnr/PROVESTIV/" + str(imgID) + ".jpg")
            #     plt.clf()
            #
            #     for k in range():
            #         # scarta gli elementi nell'xml che non si vogliono
            #         if not doc[k].clsname.startswith(clsnameBeginning):
            #             continue
            #
            #         l = doc[k].left
            #         t = doc[k].top
            #         w = doc[k].width
            #         h = doc[k].height
            #
            #         squaresColor = 0.8
            #
            #         for i in range(w):
            #             tmpImg[t][l + i] = squaresColor
            #             tmpImg[t + h][l + i] = squaresColor
            #
            #         for i in range(h):
            #             tmpImg[t + i][l] = squaresColor
            #             tmpImg[t + i][l + w] = squaresColor




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
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
