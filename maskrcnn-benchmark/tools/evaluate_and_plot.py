# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import json

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip


import matplotlib.pyplot as plt
# TODO import da rimettere sotto (messo qui perché sennò viene impostato 'agg') e usare plt.figure invece di plt.plot


import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


from bisect import bisect_left

from tqdm import tqdm

# Use parameter DATASETS.TEST to change evaluated datasets
# Example: DATASETS.TEST "('muscima_train', 'muscima_val')"
# Note: value specified must be a tuple

# Use parameter MODEL.DEVICE to run on different device
# Example: MODEL.DEVICE "cpu"

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        # default="configs/muscima/e2e_faster_rcnn_R_50_FPN_1x_muscima_pretrained_imagenet.yaml",
        default="configs/muscima/e2e_faster_rcnn_R_50_FPN_1x_muscima.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--output_dirs", type=str, default=None, help="multiple dirs, will be compared")
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

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.output_dirs is None:
        if args.config_file.startswith("configs/"):
            # se il file di configurazione specificato sta nella cartella configs, allora determino automaticamente la cartella di output
            variant_folders = ("models/" + args.config_file[len("configs/"):-len(".yaml")],)
            cfg.update({"OUTPUT_DIR": variant_folders[0]})
        else:
            # se il file di configurazione non sta nella cartella configs, allora l'output deve essere specificato mediante OUTPUT_DIR
            variant_folders = (cfg.OUTPUT_DIR,)
    else:
        # se ho specificato --output_dirs, questo parametro ha maggiore priorità rispetto ad OUTPUT_DIR e al --config-file
        # --output_dirs serve per confrontare varianti diverse
        variant_folders = args.output_dirs.split(",")

    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    metrics_comparison_datasets = {}
    metrics_comparison_variants = {}

    for jdx, variant_folder in enumerate(variant_folders):

        cfg['OUTPUT_DIR'] = variant_folder
        pthFiles = [f for f in os.listdir(variant_folder) if os.path.isfile(os.path.join(variant_folder, f)) and f.endswith(".pth")]
        assert all(pthFile.endswith(".pth") for pthFile in pthFiles)

        for pthFile in pthFiles:
            pthPrefix = pthFile[:-4]
            cfg['MODEL'].update({"WEIGHT": os.path.join(variant_folder, pthFile)})
            model = build_detection_model(cfg)
            model.to(cfg.MODEL.DEVICE)

            output_dir = cfg.OUTPUT_DIR
            checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
            _ = checkpointer.load(cfg.MODEL.WEIGHT)

            iou_types = ("bbox",)
            if cfg.MODEL.MASK_ON:
                iou_types = iou_types + ("segm",)
            output_folders = [None] * len(cfg.DATASETS.TEST)
            dataset_names = cfg.DATASETS.TEST
            if cfg.OUTPUT_DIR:
                for idx, dataset_name in enumerate(dataset_names):
                    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, pthPrefix)
                    mkdir(output_folder)
                    output_folders[idx] = output_folder
            data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
            for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
                if not os.path.exists(os.path.join(output_folder, "coco_results.pth")):
                    inference(
                        model,
                        data_loader_val,
                        dataset_name=dataset_name,
                        iou_types=iou_types,
                        box_only=cfg.MODEL.RPN_ONLY,
                        device=cfg.MODEL.DEVICE,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=output_folder,
                    )
                synchronize()

            for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):

                avgDistFilePath = os.path.join(output_folder, "average_distance.pth")
                if not os.path.exists(avgDistFilePath):
                    AD = 0
                    truths = loadPredictionsFromJsonData(data_loader_val.dataset.coco.dataset['annotations'])
                    preds = loadPredictionsFromJson(os.path.join(output_folder, "bbox.json"), 0.7)

                    allkeys = set(preds.keys()).union(set(truths.keys()))
                    for image in allkeys:
                        if image not in truths or image not in preds:
                            AD += 1
                            continue
                        trueSeq = getLabelsSequence(*sortAnnotations(truths[image]["bboxes"], truths[image]["labels"]))
                        predSeq = getLabelsSequence(*sortAnnotations(preds[image]["bboxes"], preds[image]["labels"]))
                        AD += normalizedSequencesDistance(trueSeq, predSeq)
                    AD /= len(allkeys)
                    torch.save(AD, avgDistFilePath)

                AD = torch.load(avgDistFilePath)
                results = torch.load(os.path.join(output_folder, "coco_results.pth")).results['bbox']
                results["AD"] = AD
                for metric in results:
                    if metric not in metrics_comparison_datasets:
                        metrics_comparison_datasets[metric] = {}
                    if dataset_name not in metrics_comparison_datasets[metric]:
                        metrics_comparison_datasets[metric][dataset_name] = {}
                    metrics_comparison_datasets[metric][dataset_name][pthPrefix] = results[metric]

                    if dataset_name not in metrics_comparison_variants:
                        metrics_comparison_variants[dataset_name] = {}
                    if metric not in metrics_comparison_variants[dataset_name]:
                        metrics_comparison_variants[dataset_name][metric] = {}
                    if variant_folder not in metrics_comparison_variants[dataset_name][metric]:
                        metrics_comparison_variants[dataset_name][metric][variant_folder] = {}
                    metrics_comparison_variants[dataset_name][metric][variant_folder][pthPrefix] = results[metric]

        plot_comparison_datasets(metrics_comparison_datasets, os.path.join(cfg.OUTPUT_DIR, "inference"))

    plot_comparison_variants(metrics_comparison_variants, "inference")


# crea grafici e li salva in vari file .png
def plot_comparison_datasets(metrics, path):
    # per ogni metrica faccio un grafico
    for metric in metrics:
        plt.title(metric)
        # per ogni dataset faccio una funzione
        for dataset_name in metrics[metric]:
            pointsX = []
            pointsY = []
            # per ogni file .pth ho un punto nella funzione
            for pthPrefix in metrics[metric][dataset_name]:
                x = int(pthPrefix[-7:])             # la x del punto è l'iterazione
                # inserisco il punto in modo che pointsX sia ordinato per x crescente
                index = bisect_left(pointsX, x)
                pointsX.insert(index, x)
                pointsY.insert(index, metrics[metric][dataset_name][pthPrefix])

            plt.plot(pointsX, pointsY, label=dataset_name)

        plt.legend(loc='best')
        if not os.path.exists(path):
            os.makedirs(path)
        imgFile = os.path.join(path, metric + ".png")
        plt.savefig(imgFile)
        plt.clf()
        print("plot comparison of datasets saved to: " + imgFile)


def plot_comparison_variants(metrics, path):
    # per ogni dataset
    for dataset_name in metrics:
        # per ogni metrica faccio un grafico
        for metric in metrics[dataset_name]:
            plt.title(metric)
            # per ogni variante faccio una funzione
            for variant in metrics[dataset_name][metric]:
                pointsX = []
                pointsY = []
                # per ogni file .pth ho un punto nella funzione
                for pthPrefix in metrics[dataset_name][metric][variant]:
                    x = int(pthPrefix[-7:])             # la x del punto è l'iterazione
                    # inserisco il punto in modo che pointsX sia ordinato per x crescente
                    index = bisect_left(pointsX, x)
                    pointsX.insert(index, x)
                    pointsY.insert(index, metrics[dataset_name][metric][variant][pthPrefix])

                variantLabel = variant[variant.rindex("/")+1:]
                plt.plot(pointsX, pointsY, label=variantLabel)

            plt.legend(loc='best')
            imgDir = os.path.join(path, dataset_name)
            if not os.path.exists(imgDir):
                os.makedirs(imgDir)
            imgFile = os.path.join(imgDir, metric + ".png")
            plt.savefig(imgFile)
            plt.clf()
            print("plot comparison of variants saved to: " + imgFile)


# ordina le annotazioni per xmin crescente
def sortAnnotations(bboxes, labels):
    bboxes, labels = [list(z) for z in
                              zip(*sorted(zip(bboxes, labels), key=lambda l: l[0][0]))]
    return bboxes, labels


# ritorna una lista di liste di label: la lista esterna scandisce il tempo, ogni lista interna contiene le note suonate in un determinato istante (infatti possono esserci più note in contemporanea)
def getLabelsSequence(bboxes, labels):
    # l'ordinamento deve già essere stato fatto

    assert len(bboxes) == len(labels)
    if len(bboxes) == 0:
        return []

    # (i bbox sono in formato x1, y1, x2, y1)

    sequence = []  # lista di liste di label
    bbox = [-1, -1, -1, -1]  # bbox fasullo, è comodo per non gestire la prima iterazione del ciclo con un if (perché non esiste un bbox precedente al primo)
    for i in range(len(bboxes)):
        previousBbox = bbox
        bbox = bboxes[i]
        label = labels[i]
        if bbox[0] <= previousBbox[2]:  # bbox.x1 <= previousBbox.x2
            # se questa annotazione sta sopra o sotto all'annotazione precedente (ovvero le note sono da suonare nello stesso istante)
            sequence[-1].append(label)
        else:
            # se l'annotazione i-esima è lontana da quella precedente, metto la label in un nuovo istante di tempo
            sequence.append([label])

    return sequence


cache = {}
# ritorna la distanza tra due sequenze, non normalizzata
def sequencesDistance(trueSeq, predSeq):
    if len(trueSeq) < len(predSeq):
        trueSeq, predSeq = predSeq, trueSeq

    seqsStr = str(trueSeq) + str(predSeq)
    if seqsStr in cache:
        return cache[seqsStr]

    if len(predSeq) == 0:
        cache[seqsStr] = len([label for instant in trueSeq for label in instant])
        return cache[seqsStr]

    t = set(trueSeq[-1])  # insieme delle note nell'ultimo istante del groundtruth
    p = set(predSeq[-1])  # insieme delle note nell'ultimo istante della predizione
    cost = len(t - p) + len(p - t)  # numero di elementi per cui i due insiemi differiscono

    # ricorsione edit-distance
    cache[seqsStr] = min([sequencesDistance(trueSeq[:-1], predSeq) + len(t),
                          sequencesDistance(trueSeq, predSeq[:-1]) + len(p),
                          sequencesDistance(trueSeq[:-1], predSeq[:-1]) + cost])
    return cache[seqsStr]


def normalizedSequencesDistance(trueSeq, predSeq):
    n = max(len([label for instant in trueSeq for label in instant]), len([label for instant in predSeq for label in instant]))
    return sequencesDistance(trueSeq, predSeq) / n


def loadPredictionsFromJsonData(js, th=None):
    preds = {}
    for ann in js:
        if th is not None and ann['score'] < th:
            continue
        an_id = ann['image_id']
        if an_id in preds:
            preds[an_id]['bboxes'].append(ann['bbox'])
            preds[an_id]['labels'].append(ann['category_id'])
        else:
            preds[an_id] = {}
            preds[an_id]['bboxes'] = [ann['bbox']]
            preds[an_id]['labels'] = [ann['category_id']]

    return preds


def loadPredictionsFromJson(jsonPath, th=0.7):
    with open(jsonPath, "r") as jsonFile:
        js = json.load(jsonFile)
    return loadPredictionsFromJsonData(js, th)


if __name__ == "__main__":
    main()
