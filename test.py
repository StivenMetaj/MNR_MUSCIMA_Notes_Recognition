import copy
import os
from muscima.io import parse_cropobject_list

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# TODO ordine note
# TODO in quale rigo o spazio si trova la nota


def getPentasSeparators(sums):
    th = np.max(sums) / 2
    peak = False
    count = 0
    cuts = []
    previousEnd = 0
    y = 0
    for s in sums:
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

def getPentasLimits(sums):
    th = np.max(sums) / 2
    peak = False
    count = 0
    limits = []
    y = 0
    for s in sums:
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

def squarize(doc, img, clsnameBeginning=""):

    tmpImg = copy.deepcopy(img)

    for k in range(len(doc)):
        if not doc[k].clsname.startswith(clsnameBeginning):
            continue

        if k==10:
            break

        l = doc[k].left
        t = doc[k].top
        w = doc[k].width
        h = doc[k].height

        for i in range(w):
            tmpImg[t][l+i] = 0.8
            tmpImg[t+h][l+i] = 0.8

        for i in range(h):
            tmpImg[t+i][l] = 0.8
            tmpImg[t+i][l+w] = 0.8

    return tmpImg


def getOrderedNotesAnnotations(doc, cuts):

    notes = []
    for i in range(len(cuts) - 1):
        notes.append([])

    for k in range(len(doc)):
        if not doc[k].clsname.startswith("notehead"):
            continue

        # guardo in quale lista devo appendere la nota
        yNoteheadCenter = doc[k].top + (doc[k].height/2)
        i = 0
        while yNoteheadCenter > cuts[i]:
            i += 1
        # appendo la nota alla lista giusta
        notes[i-1].append(doc[k])

    # ordino le liste
    for i in range(len(notes)):
        notes[i] = sorted(notes[i], key=lambda x: x.left)

    return notes



def preprocessStaffLines(imgStaff, numLines):
    img = copy.deepcopy(imgStaff)

    yExploredLines = []
    for x in range(img.shape[1]):
        for y in range(len(img)):
            flag = True
            for ey in yExploredLines:
                if abs(ey - y) < 4:
                    flag = False
                    break

            if img[y][x] == 1.0 and flag:
                numLines -= 1
                yExploredLines.append(y)
                yCur = y
                for k in range(x+1, img.shape[1]):
                    if img[yCur][k] == 1.0:
                        continue
                    elif img[yCur-1][k] == 1.0:
                        yCur -= 1
                    elif img[yCur+1][k] == 1.0:
                        yCur += 1
                    else:
                        img[yCur][k] = 1.0

                if numLines == 0:
                    return img

    return "Error: not enough lines found"

def addLedgers(imgStaff, doc):
    img = copy.deepcopy(imgStaff)

    for annotation in doc:
        if not annotation.clsname.startswith("ledge"):
            continue
        y = annotation.top + int(annotation.height/2)
        for x in range(annotation.width):
            img[y][annotation.left + x] = 1.0

    return img

def addNotesCenter(imgStaff, doc):
    img = copy.deepcopy(imgStaff)

    for annotation in doc:
        if not annotation.clsname.startswith("notehead"):
            continue
        y = annotation.top + int(annotation.height/2)
        x = annotation.left + int(annotation.width/2)
        img[y][x] = 0.654321

    return img


def getNotesPositions(imgStaff, imgStaffLedgers, notesAnnotations, limits):

    notesPositions = []
    stopValue = ((limits[0][1] - limits[0][0])/4) * 1.5

    for annotatedStaff in notesAnnotations:
        notesPos = []
        for annotation in annotatedStaff:
            if isInsideStaff(annotation, limits):
                #notePos = insideStaffPosition(imgStaff, annotation, stopValue)
                u, l = insideStaffPosition(imgStaff, annotation, stopValue)
                if u == 5 and l == 0:
                    pass
            # TODO gestire fuori pentagramma
            # else:
            #     notesPos = ...
            # notesPos.append(notePos)
        notesPositions.append(notesPos)

    return notesPositions


def isInsideStaff(annotation, limits):
    # TODO cambiare/aggiungere implementazione (guardando XML)
    yAnnotationCenter = annotation.top + int(annotation.height/2)
    eps = int(annotation.height*3/4)

    for limit in limits:
        if limit[0] - eps < yAnnotationCenter < limit[1] + eps:
            return True

    return False

def insideStaffPosition(imgStaff, annotation, stopValue):

    yAnnotationCenter = annotation.top + int(annotation.height/2)
    xAnnotationCenter = annotation.left + int(annotation.width/2)

    numBlacks = 0
    upperLines = 0
    lastBlack = True
    i = int(annotation.height/3)
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
    i = int(annotation.height/3)
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

def outsideStaffPosition(imgStaff, annotation, stopValue):
    pass

# TODO capire differenza tra i due XML
CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_manual'
# CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_withstaff'

cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]

# giusto per far prima
docsNumber = 5
cropobject_fnames = cropobject_fnames[:docsNumber]

docs = [parse_cropobject_list(f) for f in cropobject_fnames]

for docID in range(docsNumber):

    doc = docs[docID]

    print(doc[0].uid)

    w = doc[0].uid[31:33]
    p = doc[0].uid[36:38]
    
    imgPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/image/p0" + p + ".png"
    imgStaffPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/gt/p0" + p + ".png"

    #img = mpimg.imread(imgPath)
    imgStaff = mpimg.imread(imgStaffPath)

    horizontalProjection = np.sum(imgStaff, axis=1)

    pentasSeparators = getPentasSeparators(horizontalProjection)
    pentasLimits = getPentasLimits(horizontalProjection)


    notesAnnotations = getOrderedNotesAnnotations(doc, pentasSeparators)

    print(notesAnnotations)
    
    # sums = sums.tolist()
    # 
    # plt.plot(sums)
    # plt.show()

    # TODO preprocess staff: completare staff + aggiunta ledger
    imgStaff = preprocessStaffLines(imgStaff, (len(pentasSeparators) - 1) * 5)
    imgStaffLedgers = addLedgers(imgStaff, doc)
    #imgStaff = addNotesCenter(imgStaff, doc)


    plt.imshow(imgStaff, cmap="gray")
    plt.show()

    # TODO trasforma annotazioni in informazione sulla posizione
    notesPositions = getNotesPositions(imgStaff, imgStaffLedgers, notesAnnotations, pentasLimits)
    

exit(0)

print(imgStaffPath)

print(len(docs[0]))

# 0,0
# 766,1860
# 767,1860
print("img(0,0): " + str(imgStaff[0][0]))
print("img(766,1860): " + str(imgStaff[766][1860]))
print("img(767,1860): " + str(imgStaff[767][1860]))
