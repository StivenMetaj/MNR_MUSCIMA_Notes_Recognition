import copy
import os
import random

import cv2
from muscima.io import parse_cropobject_list

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
    th = np.max(horizontalProjection) / 2
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

    cuts.append(int((y + previousEnd) / 2))
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
def getOrderedNotesAnnotations(doc, pentasSeparators):
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
            # TODO gestire fuori pentagramma (funzione outside vuota)
            else:
                u, l = outsideStaffPosition(imgStaff, note, stopValue)
            notesPos.append((u, l))
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

#Ritorna le coordinate y e x della nota
def getNoteCoordinates(noteAnnotation):
    return noteAnnotation.top + int(noteAnnotation.height / 2), noteAnnotation.left + int(noteAnnotation.width / 2)

#Ritorna la posizione della nota all'interno del pentagramma, nel caso in cui la nota sia fuori dalle staff
#TODO
def outsideStaffPosition(imgStaff, annotation, stopValue):
    return 1, 1

#Ritorna una lista di immagini contenenti solo le staffs, con i rispettivi OFFSET sulle y
#In input si ha l'immagine di partenza
def getStaffsFromImage(img):
    horizontalProjection = np.sum(img, axis=1)
    pentasSeparators = getPentasSeparators(horizontalProjection)
    staffsAndOffsetsY = []

    #per separatori - 1 volte taglio l'immagine di partenza e la salvo in staff
    for i in range(len(pentasSeparators) - 1):
        start, end = pentasSeparators[i], pentasSeparators[i + 1]
        staff = img[start: end]
        staffsAndOffsetsY.append((staff, (start, end)))

    return staffsAndOffsetsY

#Ritorna il resize dell'immagine 'patch' secondo le dimensioni w h
def resizePatch(patch, w, h):
    return cv2.resize(patch, dsize=(w, h), interpolation=cv2.INTER_AREA)

#Ritorna la misura dopo aver fatto un resize da startDim a finalDim
#Si e' pensato al fatto che il resize è operazione lineare e quindi il calcolo deriva dalla
#proporzione [start : final = measure : X]
#In input si hanno quindi i 3 valori
def getResizedMeasure(startDimension, finalDimension, measure):
    return (finalDimension*measure)/startDimension

#Ritorna una lista di immagini contenenti solo le patch per il dataset
#In input si hanno l'immagine del pentagramma singolo e le dimensioni w, h
def getPatchesFromStaff(staff, w, h):
    patchesAndOffsetsX = []
    l = len(staff)

    transpStaff = staff.transpose()
    # numero di patch sequenziali
    n = int(staff.shape[1] / l)
    for i in range(n):
        #taglio il pentagramma in modo sequenziale
        start, end = i * l, (i + 1) * l
        patch = transpStaff[start: end].transpose()
        # print(str(i*l))
        # plt.imshow(patch, cmap="gray")
        # plt.show()
        patch = resizePatch(patch, w, h)
        patchesAndOffsetsX.append((patch, (start, end)))


        #creo patch casuali di ugual numero a quelle sequenziali
        start = random.randint(0, staff.shape[1] - l)
        end = start + l
        patch = transpStaff[start: end].transpose()
        patch = resizePatch(patch, w, h)
        patchesAndOffsetsX.append((patch, (start, end)))

    return patchesAndOffsetsX

#Funzione che cicla su tutti i file xml
def ciclone():
    # TODO capire differenza tra i due XML
    CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_manual'
    # CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_withstaff'

    cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
    #per debuggare
    cropobject_fnames = cropobject_fnames[:5]

    docs = [parse_cropobject_list(f) for f in cropobject_fnames]

    for docID in range(len(docs)):
        doc = docs[docID]

        #prendo dall'xml l'id del writer (w) e dello spartito (p)
        w = doc[0].uid[31:33]
        p = doc[0].uid[36:38]

        imgPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/image/p0" + p + ".png"
        imgStaffPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/gt/p0" + p + ".png"

        imgStaff = mpimg.imread(imgStaffPath)

        horizontalProjection = np.sum(imgStaff, axis=1)

        pentasSeparators = getPentasSeparators(horizontalProjection)

        notesAnnotations = getOrderedNotesAnnotations(doc, pentasSeparators)

        #per plottare la proiezione orizzontale si deve prima trasformare il vettore in una lista
        # plt.plot(sums.tolist())
        # plt.show()

        imgStaff = getPreprocessedStaffImage(imgStaff)
        imgStaffLedgers = getStaffImageWithLedgers(imgStaff, doc)

        plt.imshow(imgStaff, cmap="gray")
        plt.show()

        notesPositions = getNotesPentasPositions(imgStaff, imgStaffLedgers, notesAnnotations)
        print(notesPositions)

#funzione che taglia le immagini in patches
def funzioneMegaPerLePatches(w, p, dimX, dimY):
    imgPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/image/p0" + p + ".png"
    imgStaffPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/gt/p0" + p + ".png"

    img = mpimg.imread(imgPath)

    staffs = getStaffsFromImage(img)

    for staff in staffs:
        patches = getPatchesFromStaff(staff[0], dimX, dimY)
        for patch in patches:
            plt.imshow(patch[0], cmap="gray")
            plt.show()

# TODO aggiungere notazioni a patches... continuare la cosa!
# TODO funzione che prende (upper, lower) e restituisce la classe, ovvero un intero appartenente a [-5, 5]
if __name__ == "__main__":
    prova = 1
    if prova == 1:
        funzioneMegaPerLePatches("01", "10", 128, 128)
    if prova == 2:
        ciclone()
