import matplotlib
matplotlib.use("Agg")
#import matplotlib.image as mpimg
from PIL import Image

import os
import torch
import torch.utils.data
from torch import Tensor
import numpy as np

import sys

from xml.dom import minidom

from maskrcnn_benchmark.structures.bounding_box import BoxList

# questa classe adatta il dataset alla libreria facebookresearch/maskrcnn-benchmark
# questa classe poteva forse essere evitata (visto che il formato è lo stesso di PASCAL VOC), però per adesso l'ho messa
class MuscimaDataset(torch.utils.data.Dataset):

    # TODO testare se funziona con la libreria facebookresearch/maskrcnn-benchmark (anche con solo pochi elementi del dataset)

    # numExamples serve solo per prendere un sottoinsieme del dataset (per fare le prove più velocemente): se <=0, viene preso tutto il dataset
    def __init__(self, data_dir, split, use_difficult=False, transforms=None, numExamples=50):

        #data_dir = "maskrcnn-benchmark/" + data_dir

        imagesDir = os.path.join(data_dir, "JPEGImages")
        annotationsDir = os.path.join(data_dir, "Annotations")

        # leggo tutte le immagini del dataset e le metto in un Tensor, facendo la stessa cosa per le annotazioni
        self.dataset = []
        self.labels = []
        self.boxes = []
        self.exampleNumbers = [] #giusto per controllare che l'associazione tra immagini e annotazioni avvenga correttamente TODO rimuovere
        self.transforms = transforms

        for jpgFile in os.listdir(imagesDir):
            jpg_path = os.path.join(imagesDir, jpgFile)
            if jpgFile.endswith("jpg"):
                # leggo l'immagine
                #image = mpimg.imread(jpg_path)
                image = Image.open(jpg_path).convert("RGB")

                # controllo che esista l'annotazione corrispondente all'immagine
                xmlFile = jpgFile[:-3] + "xml"
                xmlPath = os.path.join(annotationsDir, xmlFile)
                assert os.path.isfile(xmlPath)

                # leggo il file csv per leggere le annotazioni
                parsedXML = minidom.parse(xmlPath)

                imageLabel = []   # questa lista conterrà solo le annotazioni di jpgFile
                imageBoxes = []         # questa lista conterrà solo i box di jpgFile
                for annotation in parsedXML.getElementsByTagName("name"):
                    label = annotation.firstChild.data

                    if label == "OutOfStaffs":   # per ora non considero le note fuori dal pentagramma
                        # TODO gestire note fuori dal pentagramma
                        continue

                    bndbox = annotation.nextSibling.nextSibling
                    assert bndbox.nodeName == "bndbox"

                    box = [int(bndbox.getElementsByTagName("xmin")[0].firstChild.data),  # x1
                           int(bndbox.getElementsByTagName("ymin")[0].firstChild.data),  # y1
                           int(bndbox.getElementsByTagName("xmax")[0].firstChild.data),  # x2
                           int(bndbox.getElementsByTagName("ymax")[0].firstChild.data)]  # y2

                    label = int(label) # eccezione volutamente non gestita: non voglio che l'xml contenga roba strana
                    imageLabel.append(label)
                    imageBoxes.append(box)

                if len(imageBoxes) == 0:
                    continue

                # aggiungo l'immagine al dataset
                self.dataset.append(image)
                # aggiungo le annotazioni relativi all'immagine alla lista di annotazioni
                self.labels.append(imageLabel)
                self.boxes.append(imageBoxes)
                self.exampleNumbers.append(jpgFile[:-4])    #TODO rimuovere

                numExamples -= 1
                if numExamples == 0:
                    break

        #self.dataset = Tensor(np.array(self.dataset))
        self.labels = self.labels
        # non posso trasformare labels in un Tensor subito perché le img hanno un num variabile di annotazioni



    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = self.dataset[idx]

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        #boxes = [[0, 0, 10, 10], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50]]
        boxes = self.boxes[idx]
        # and labels
        #labels = torch.tensor([10, 20, 10, 20, 10, 20, 10, 20, 2, 3])
        labels = Tensor(self.labels[idx])

        assert len(boxes) == len(labels)

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def __len__(self):
        return len(self.dataset)

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        # TODO numeri qui sono brutti, eventualmente cambiare, ma forse non serve nemmeno questo pezzo
        return {"height": 128, "width": 128}
