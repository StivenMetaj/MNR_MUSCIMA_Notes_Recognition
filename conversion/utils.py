import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg

import copy
import os

import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch
from torch import Tensor

from xml.dom import minidom

#from muscima.io import parse_cropobject_list

dataDir = 'MNR2019'
imagesDir = dataDir + '/JPEGImages'
annotationsDir = dataDir + '/Annotations'

# TODO fare un bel refactor e ripulire ciò che non serve

# Ritorna la proiezione orizzontale. In input si ha l'immagine target
def getHorizontalProjection(img):
    return np.sum(img, axis=1)


# Ritorna il numero di linee di una immagine di Staff. In input si ha l'immagine target
def getNumLines(imgStaff):
    horProj = getHorizontalProjection(imgStaff)
    pentasSeparators = getPentasSeparators(horProj)
    return (len(pentasSeparators) - 1) * 5


# Ritorna le posizioni dei separatori dei pentagrammi
# In input si ha la proiezione orizzontale di una immagine
def getPentasSeparators(horizontalProjection):
    th = (np.max(horizontalProjection)) / 2
    peak = False
    count = 0
    cuts = []
    previousEnd = 0
    y = 0

    for s in horizontalProjection:
        if s > th:
            if not peak and count == 0:
                # salita
                nextStart = y
                cuts.append(int((nextStart + previousEnd) / 2))
            peak = True
        else:
            if peak:
                # discesa
                count += 1
                if count == 5:
                    # fine pentagramma
                    previousEnd = y
                    count = 0
            peak = False
        y += 1

    cuts.append(y)
    cuts[0] = 0
    return cuts


# Ritorna una lista di x tuple per x pentagrammi. Ogni tupla contiene posizione di inizio e di fine del pentagramma
# In input si ha la proiezione orizzontale di una immagine
def getPentasLimits(horizontalProjection):
    th = np.max(horizontalProjection) / 2
    peak = False
    count = 0
    limits = []
    y = 0

    for s in horizontalProjection:
        if s > th:
            if not peak and count == 0:
                # salita
                start = y
            peak = True
        else:
            if peak:
                # discesa
                count += 1
                if count == 5:
                    # fine pentagramma
                    end = y
                    count = 0
                    limits.append((start, end))
            peak = False
        y += 1

    return limits


# Ritorna l'immagine (deepcopy) con l'aggiunta di quadrati a seconda del clsname che si cerca
# In input si ha l'immagine, il clsname da trovare e il documento XML di supervisione
def squarize(doc, img, clsnameBeginning=""):
    tmpImg = copy.deepcopy(img)

    for k in range(len(doc)):
        # scarta gli elementi nell'xml che non si vogliono
        if not doc[k].clsname.startswith(clsnameBeginning):
            continue

        l = doc[k].left
        t = doc[k].top
        w = doc[k].width
        h = doc[k].height

        squaresColor = 0.8

        for i in range(w):
            tmpImg[t][l + i] = squaresColor
            tmpImg[t + h][l + i] = squaresColor

        for i in range(h):
            tmpImg[t + i][l] = squaresColor
            tmpImg[t + i][l + w] = squaresColor

    return tmpImg


# Ritorna una lista di liste. Ogni sottolista contiene le note di un pentagramma
# in ordine del parametro left del XML
# In input si ha il documento XML di supervisione e la lista
def getOrderedNotesAnnotations(doc, imgStaff):
    horizontalProjection = getHorizontalProjection(imgStaff)
    pentasSeparators = getPentasSeparators(horizontalProjection)
    notes = []
    for i in range(len(pentasSeparators) - 1):
        notes.append([])

    for k in range(len(doc)):
        # scarto tutto quello che non è una nota
        if not doc[k].clsname.startswith("notehead"):
            continue

        # guardo in quale lista devo appendere la nota
        yNoteheadCenter = doc[k].top + (doc[k].height / 2)
        i = 0
        while yNoteheadCenter > pentasSeparators[i]:
            i += 1
        # appendo la nota alla lista giusta
        notes[i - 1].append(doc[k])

    # ordino le liste in base al "left"
    for i in range(len(notes)):
        notes[i] = sorted(notes[i], key=lambda x: x.left)

    return notes


# Ritorna l'immagine (deepcopy) con l'aggiunta dei pixel per avere staff continue
# In input si ha l'immagine delle staff
def getPreprocessedStaffImage(imgStaff):
    img = copy.deepcopy(imgStaff)
    numLines = getNumLines(imgStaff)

    yExploredLines = []
    for x in range(img.shape[1]):
        for y in range(len(img)):
            flag = True
            # controllo di non contare linee già processate
            for ey in yExploredLines:
                # le posizioni salvate vanno controllate con un errore (di 4 in questo caso)
                if abs(ey - y) < 4:
                    flag = False
                    break

            # se trovo un pixel allora inizio a processare la linea
            if img[y][x] == 1.0 and flag:
                numLines -= 1
                # aggiungo la posizione della linea a quelle esplorate
                yExploredLines.append(y)
                yCur = y
                # scorro la linea e se non trovo pixel li aggiungo
                for k in range(x + 1, img.shape[1]):
                    if img[yCur][k] == 1.0:
                        continue
                    elif img[yCur - 1][k] == 1.0:
                        yCur -= 1
                    elif img[yCur + 1][k] == 1.0:
                        yCur += 1
                    else:
                        img[yCur][k] = 1.0

                # se non ho più linee ritorno l'immagine
                if numLines == 0:
                    return img
    return "Error: not enough lines found"


# Ritorna l'immagine (deepcopy) con l'aggiunta dei ledgers (linee per note esterne)
# In input si ha l'immagine delle staff e il documento XML di supervisione
def getStaffImageWithLedgers(imgStaff, doc):
    img = copy.deepcopy(imgStaff)

    for annotation in doc:
        # voglio solo i ledgers
        if not annotation.clsname.startswith("ledge"):
            continue
        y = annotation.top + int(annotation.height / 2)
        for x in range(annotation.width):
            # disegno il ledger
            img[y][annotation.left + x] = 1.0
    return img


# Ritorna l'immagine (deepcopy) con l'aggiunta dei punti centrali delle note
# In input si ha l'immagine delle staff e il documento XML di supervisione
def addNotesCenter(imgStaff, doc):
    img = copy.deepcopy(imgStaff)

    for annotation in doc:
        if not annotation.clsname.startswith("notehead"):
            continue
        y = annotation.top + int(annotation.height / 2)
        x = annotation.left + int(annotation.width / 2)
        img[y][x] = 0.7

    return img


# Ritorna la lista delle posizioni all'interno del pentagramma (up e low) per ogni nota
# In input si hanno l'immagine delle staff e la lista delle note
def getNotesPentasPositions(imgStaff, imgStaffLedgers, notesAnnotations):
    notesPositions = []
    pentasLimits = getPentasLimits(getHorizontalProjection(imgStaff))
    # lo stopValue indica la distanza tra una riga e l'altra, con un eccesso del 50% (*1,5)
    stopValue = ((pentasLimits[0][1] - pentasLimits[0][0]) / 4) * 1.5

    for staff in notesAnnotations:
        # per ogni pentagramma avro' una lista notesPos
        notesPos = []
        for note in staff:
            if isInsideStaff(note, pentasLimits):
                u, l = getInsideStaffNotePosition(imgStaff, note, stopValue)
            else:
                # TODO gestire fuori pentagramma (funzione outside vuota)
                #u, l = getOutsideStaffPosition(imgStaffLedgers, note, stopValue)
                continue
            notesPos.append(l - u)

        notesPositions.append(notesPos)

    return notesPositions


# Ritorna True se la nota si trova all'interno di uno degli intervalli di limits
# In input si hanno la nota (noteAnnotation) e i limiti del pentagramma
def isInsideStaff(noteAnnotation, pentalimits):
    # TODO cambiare/aggiungere implementazione (guardando XML)
    yAnnotationCenter = noteAnnotation.top + int(noteAnnotation.height / 2)
    # con eps vado a prendere anche le note che non si trovano in modo preciso tra i limits
    # ma che comunque devono essere considerate "Inside Notes"
    eps = int(noteAnnotation.height * 3 / 4)

    for limit in pentalimits:
        if limit[0] - eps < yAnnotationCenter < limit[1] + eps:
            return True

    return False


# Ritorna la posizioe della nota all'interno del pentagramma
# in uscita si hanno 2 interi, U ed L (up e low) che indicano il numero di righe
# sopra e sotto la nota
#
# Se la somma dei numeri e' 5 la nota si trova tra due linee, altrimenti (somma=4) su una linea
# In input si hanno l'immagine delle staff, la nota e il valore per cui smetto di contare le righe
def getInsideStaffNotePosition(imgStaff, noteAnnotation, stopValue):
    yAnnotationCenter, xAnnotationCenter = getNoteCoordinates(noteAnnotation)

    numBlacks = 0
    upperLines = 0
    lastBlack = True
    i = int(noteAnnotation.height / 3)
    # cerco le upperLines
    while numBlacks < stopValue:
        if imgStaff[yAnnotationCenter - i][xAnnotationCenter] == 0.0:
            numBlacks += 1
            lastBlack = True
        elif lastBlack:
            # salita
            lastBlack = False
            upperLines += 1
            numBlacks = 0
        i += 1

    numBlacks = 0
    lowerLines = 0
    lastBlack = True
    i = int(noteAnnotation.height / 3)
    # cerco le lowerLines
    while numBlacks < stopValue:
        if imgStaff[yAnnotationCenter + i][xAnnotationCenter] == 0.0:
            numBlacks += 1
            lastBlack = True
        elif lastBlack:
            # salita
            lastBlack = False
            lowerLines += 1
            numBlacks = 0
        i += 1

    return upperLines, lowerLines


# Ritorna le coordinate y e x della nota
def getNoteCoordinates(noteAnnotation):
    return noteAnnotation.top + int(noteAnnotation.height / 2), noteAnnotation.left + int(noteAnnotation.width / 2)


# Ritorna la posizione della nota all'interno del pentagramma, nel caso in cui la nota sia fuori dalle staff
# TODO
def getOutsideStaffPosition(imgStaffLedgers, annotation, stopValue):
    pass


# questa classe adatta il dataset alla libreria facebookresearch/maskrcnn-benchmark
# questa classe poteva forse essere evitata (visto che il formato è lo stesso di PASCAL VOC), però per adesso l'ho messa
# AGGIORNAMENTO: questa classe era stata fatta per i file in formato VOC,
# ma è meglio usare il formato COCO (funziona l'inferenza), per formato COCO questa classe è inutile
class MuscimaDataset(object):

    # numExamples serve solo per prendere un sottoinsieme del dataset (per fare le prove più velocemente): se <=0, viene preso tutto il dataset
    def __init__(self, numExamples=-1):
        # leggo tutte le immagini del dataset e le metto in un Tensor, facendo la stessa cosa per le annotazioni
        self.dataset = []
        self.labels = []
        self.boxes = []

        for jpgFile in os.listdir(imagesDir):
            jpg_path = os.path.join(imagesDir, jpgFile)
            if jpgFile.endswith("jpg"):
                # leggo l'immagine
                image = mpimg.imread(jpg_path)

                # controllo che esista l'annotazione corrispondente all'immagine
                xmlFile = jpgFile[:-3] + "xml"
                xmlPath = os.path.join(annotationsDir, xmlFile)
                assert os.path.isfile(xmlPath)

                # leggo il file csv per leggere le annotazioni
                parsedXML = minidom.parse(xmlPath)

                imageLabel = []  # questa lista conterrà solo le annotazioni di jpgFile
                imageBoxes = []  # questa lista conterrà solo i box di jpgFile
                for annotation in parsedXML.getElementsByTagName("name"):
                    label = annotation.firstChild.data

                    if label == "OutOfStaffs":  # per ora non considero le note fuori dal pentagramma
                        # TODO gestire note fuori dal pentagramma
                        continue

                    bndbox = annotation.nextSibling.nextSibling
                    assert bndbox.nodeName == "bndbox"

                    box = [int(bndbox.getElementsByTagName("xmin")[0].firstChild.data),  # x1
                           int(bndbox.getElementsByTagName("ymin")[0].firstChild.data),  # y1
                           int(bndbox.getElementsByTagName("xmax")[0].firstChild.data),  # x2
                           int(bndbox.getElementsByTagName("ymax")[0].firstChild.data)]  # y2

                    label = int(label)  # eccezione volutamente non gestita: non voglio che l'xml contenga roba strana
                    imageLabel.append(label)
                    imageBoxes.append(box)

                if len(imageBoxes) == 0:
                    continue

                # aggiungo l'immagine al dataset
                self.dataset.append(image)
                # aggiungo le annotazioni relativi all'immagine alla lista di annotazioni
                self.labels.append(imageLabel)
                self.boxes.append(imageBoxes)

                numExamples -= 1
                if numExamples == 0:
                    break

        self.dataset = Tensor(np.array(self.dataset))
        self.labels = self.labels
        # non posso trasformare labels in un Tensor subito perché le img hanno un num variabile di annotazioni

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = self.dataset[idx]

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # boxes = [[0, 0, 10, 10], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50], [10, 20, 50, 50]]
        boxes = self.boxes[idx]
        # and labels
        # labels = torch.tensor([10, 20, 10, 20, 10, 20, 10, 20, 2, 3])
        labels = Tensor(self.labels[idx])

        assert len(boxes) == len(labels)

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        # if self.transforms:
        #     image, boxlist = self.transforms(image, boxlist).

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        # TODO numeri qui sono brutti, eventualmente cambiare, ma forse non serve nemmeno questo pezzo
        return {"height": 128, "width": 128}


if __name__ == "__main__":
    # qualche prova TODO ripulire

    # carico il dataset
    md = MuscimaDataset(numExamples=10)

    # prendo un elemento a caso del dataset
    ex = md[0]

    # non ci sono errori
    print("Example read without any error")
