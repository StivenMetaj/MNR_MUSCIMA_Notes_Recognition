# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

import argparse

from maskrcnn_benchmark.config import cfg


class MuscimaDemo(object):
    # Muscima categories for pretty print
    # CATEGORIES = [
    #     "__background__",
    #     "under_staffs",
    #     "first_line",
    #     "first_space",
    #     "second_line",
    #     "second_space",
    #     "third_line",
    #     "third_space",
    #     "fourth_line",
    #     "fourth_space",
    #     "fifth_line",
    #     "above_staffs",
    # ]

    CATEGORIES = [    # scommentare queste righe se si vuole avere delle etichette più compatte
        "__background__",
        "US",
        "L1",
        "S1",
        "L2",
        "S2",
        "L3",
        "S3",
        "L4",
        "S4",
        "L5",
        "AS"
    ]

    def __init__(
            self,
            cfg,
            confidence_threshold=0.7,
            show_mask_heatmaps=False,
            masks_per_dim=2,
            min_image_size=128,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image, returnedPredictions=None):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        # prendo le predizioni per ritornarle
        if returnedPredictions is not None:
            assert isinstance(returnedPredictions, list)
            returnedPredictions.clear()
            returnedPredictions.append(top_predictions.bbox)
            returnedPredictions.append(top_predictions.get_field('labels'))
            returnedPredictions.append(top_predictions.get_field('scores'))

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    # def overlay_mask(self, image, predictions):
    #     """
    #     Adds the instances contours for each predicted object.
    #     Each label has a different color.
    #
    #     Arguments:
    #         image (np.ndarray): an image as returned by OpenCV
    #         predictions (BoxList): the result of the computation by the model.
    #             It should contain the field `mask` and `labels`.
    #     """
    #     masks = predictions.get_field("mask").numpy()
    #     labels = predictions.get_field("labels")
    #
    #     colors = self.compute_colors_for_labels(labels).tolist()
    #
    #     for mask, color in zip(masks, colors):
    #         thresh = mask[0, :, :, None]
    #         contours, hierarchy = cv2_util.findContours(
    #             thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    #         )
    #         image = cv2.drawContours(image, contours, -1, color, 3)
    #
    #     composite = image
    #
    #     return composite

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            phase = 2
            if y >= phase:
                y -= phase
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

    # legge l'immagine specificata, calcola le predizioni su di essa e le visualizza
    def visualizePredictions(self, imgPath, prePredictionsSize, displaySize):
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        # cv2.imshow('image', img)  # scommentare se si vuole vedere l'immagine originale
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = cv2.resize(img, (prePredictionsSize, prePredictionsSize))
        returnedPredictions = []  # questa lista viene riempita da run_on_opencv_image con le predizioni
        composite = self.run_on_opencv_image(img, returnedPredictions)

        # stampo lista dei bbox e labels predetti, con relativi score
        print("Predictions for image " + str(imgPath))
        bboxes, labels, scores = returnedPredictions
        if len(bboxes) > 0:
            bboxes, labels, scores = sortAnnotations(bboxes, labels, scores)
            predSeq = getLabelsSequence(bboxes, labels)
            print("bboxes: " + str(bboxes))
            print("labels: " + str(labels))
            print("scores: " + str(scores))
            print("sequence: " + str(predSeq))

            # TODO prendere bounding box e label dal groundtruth, da essi ottenere la sequenza vera, e mostrare la distanza con quella predetta
            # ...
            # trueSeq = getLabelsSequence(... )
            # print("sequence distance from groundtruth: " + sequencesDistance(trueSeq, predSeq))
        else:
            print("no predictions with score above threshold")
        print()

        composite = cv2.resize(composite, (displaySize, displaySize))
        # i bbox si riferiscono sempre all'immagine di dimensione prePredictionsSize x prePredictionsSize
        cv2.imshow("MUSCIMA detections", composite)

        # press ESC to close current image
        while cv2.waitKey(1) != 27:
            pass
        cv2.destroyAllWindows()


# ordina le annotazioni per xmin crescente
def sortAnnotations(bboxes, labels, scores):
    bboxes, labels, scores = [torch.tensor(z) for z in
                              zip(*sorted(zip(bboxes.tolist(), labels, scores), key=lambda l: l[0][0]))]
    return bboxes, labels, scores


# ritorna una lista di liste di label: la lista esterna scandisce il tempo, ogni lista interna contiene le note suonate in un determinato istante (infatti possono esserci più note in contemporanea)
def getLabelsSequence(bboxes, labels):
    # l'ordinamento deve già essere stato fatto

    assert len(bboxes) == len(labels)
    if len(bboxes) == 0:
        return []

    # trasformo da torch.Tensor a list
    bboxes = bboxes.tolist()
    labels = labels.tolist()
    # (i bbox sono in formato x1, y1, x2, y1)

    sequence = []  # lista di liste di label
    bbox = [-1, -1, -1,
            -1]  # bbox fasullo, è comodo per non gestire la prima iterazione del ciclo con un if (perché non esiste un bbox precedente al primo)
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


# ritorna la distanza tra due sequenze, non normalizzata
def sequencesDistance(trueSeq, predSeq):
    if len(trueSeq) < len(predSeq):
        trueSeq, predSeq = predSeq, trueSeq

    if len(predSeq) == 0:
        return len([label for instant in trueSeq for label in instant])

    t = set(trueSeq[-1])  # insieme delle note nell'ultimo istante del groundtruth
    p = set(predSeq[-1])  # insieme delle note nell'ultimo istante della predizione
    cost = len(t - p) + len(p - t)  # numero di elementi per cui i due insiemi differiscono

    # ricorsione edit-distance
    return min([sequencesDistance(trueSeq[:-1], predSeq) + len(t),
                sequencesDistance(trueSeq, predSeq[:-1]) + len(p),
                sequencesDistance(trueSeq[:-1], predSeq[:-1]) + cost])


def normalizedSequencesDistance(trueSeq, predSeq):
    return sequencesDistance(trueSeq, predSeq) / len([label for instant in trueSeq for label in instant])


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        # default="../configs/muscima/e2e_faster_rcnn_R_50_FPN_1x_muscima.yaml",
        default="../configs/muscima/e2e_faster_rcnn_R_50_FPN_1x_muscima_pretrained_imagenet.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,  # TODO questo parametro è importante, va scelto bene
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=128,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 128, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    muscima_demo = MuscimaDemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    prePredictionsSize = 768  # l'immagine viene riscalata a questa dimensione prima di fare le predizioni,
    # può incidere sulla qualità delle predizioni;
    # se messo a 128 (quindi niente ridimensionamento), le annotazioni si vedono male

    displaySize = 768  # dimensione dell'immagine visualizzata a schermo, non incide sulle predizioni
    
    imgs = ['../datasets/mnr/train2019/000000000003.jpg',
            '../datasets/mnr/train2019/000000000009.jpg',
            '../datasets/mnr/train2019/000000000013.jpg',
            '../datasets/mnr/train2019/000000000028.jpg',
            '../datasets/mnr/train2019/000000000029.jpg',
            '../datasets/mnr/train2019/000000000030.jpg',
            '../datasets/mnr/train2019/000000000031.jpg',
            '../datasets/mnr/train2019/000000000032.jpg',
            '../datasets/mnr/train2019/000000000050.jpg',
            '../datasets/mnr/train2019/000000000055.jpg',
            '../datasets/mnr/train2019/000000000059.jpg',
            '../datasets/mnr/train2019/000000000060.jpg',
            '../datasets/mnr/train2019/000000000079.jpg',
            '../datasets/mnr/train2019/000000000179.jpg',
            '../datasets/mnr/train2019/000000001079.jpg',
            '../datasets/mnr/val2019/000000012798.jpg',
            '../datasets/mnr/val2019/000000012909.jpg',
            '../datasets/mnr/val2019/000000013251.jpg',
            '../datasets/mnr/val2019/000000013300.jpg',
            '../datasets/mnr/val2019/000000013301.jpg',
            '../datasets/mnr/val2019/000000013310.jpg',
            '../datasets/mnr/val2019/000000013320.jpg',
            '../datasets/mnr/val2019/000000013323.jpg',
            '../datasets/mnr/val2019/000000013400.jpg',
            '../datasets/mnr/test2019/000000017023.jpg',
            '../datasets/mnr/test2019/000000017123.jpg',
            '../datasets/mnr/test2019/000000017150.jpg',
            '../datasets/mnr/test2019/000000017223.jpg',
            '../datasets/mnr/test2019/000000017224.jpg',
            '../datasets/mnr/test2019/000000017225.jpg',
            '../datasets/mnr/test2019/000000017226.jpg',
            '../datasets/mnr/test2019/000000017227.jpg',
            '../datasets/mnr/test2019/000000017250.jpg',
            '../datasets/mnr/test2019/000000017400.jpg']

    # evaluate and show predictions for some random images
    # to close an image, press any key (don't close with mouse)
    for img in imgs:
        muscima_demo.visualizePredictions(img, prePredictionsSize, displaySize)


if __name__ == "__main__":
    # qualche test per sequencesDistance TODO rimuovere
    '''
    trueSeq = [[8, 10], [9, 7], [6, 8]]
    predSeq = [[10], [7, 9], [5], [6, 8]]

    assert sequencesDistance(trueSeq, predSeq) == 2

    s1 = [[1, 2], [3], [7, 8, 9]]
    s2 = [[2, 1], [3], [9, 7, 8]]

    assert sequencesDistance(s1, s2) == 0

    s3 = [[1, 2], [3], [7, 8], [9]]

    assert sequencesDistance(s1, s3) == 2
    assert sequencesDistance(s2, s3) == 2

    s4 = [[2], [2, 1], [3], [9, 8], [10, 11]]
    assert sequencesDistance(s1, s4) == 4
    assert sequencesDistance(s2, s4) == 4

    strangeList = [[1], [7, 6], [5], [6, 3], [10], [5], [2, 8], [10], [7, 1, 8, 4, 9], [5], [6, 8]]
    assert sequencesDistance(s1, strangeList) == sequencesDistance(strangeList, s1)
    
    print("OK")
    '''

    main()
