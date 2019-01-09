import copy
import os
from muscima.io import parse_cropobject_list

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# TODO ordine note
# TODO in quale rigo o spazio si trova la nota


def getCuts(sums):
    th = np.max(sums) / 2
    peak = False
    count = 0
    cuts = []
    previousEnd = 0
    i = 0
    for s in sums:
        if s > th:
            if not peak and count == 0:
                # salita
                nextStart = i
                cuts.append(int((nextStart + previousEnd) / 2))
            peak = True
        else:
            if peak:
                # discesa
                count += 1
                if count == 5:
                    # fine pentagramma
                    previousEnd = i
                    count = 0
            peak = False

        i += 1

    cuts.append(int((i + previousEnd) / 2))

    return cuts


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


def getOrderedNotes(doc, cuts):

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



# TODO capire differenza tra i due XML
CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_manual'
#CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_withstaff'

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

    img_path = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/image/p0" + p + ".png"
    #img_path = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/gt/p0" + p + ".png"

    img = mpimg.imread(img_path)

    sums = np.sum(img, axis=1)

    cuts = getCuts(sums)

    notes = getOrderedNotes(doc, cuts)

    print(notes)
    
    sums = sums.tolist()

    # x_plot = np.linspace(np.min(sums), np.max(sums), 300).reshape(-1, 1)
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(sums)
    # log_dens = kde.score_samples(x_plot)
    # plt.plot(x_plot, np.exp(log_dens))
    plt.plot(sums)
    plt.show()
    continue

    plt.imshow(img, cmap="gray")
    plt.show()

exit(0)

print(img_path)

print(len(docs[0]))

# 0,0
# 766,1860
# 767,1860
print("img(0,0): " + str(img[0][0]))
print("img(766,1860): " + str(img[766][1860]))
print("img(767,1860): " + str(img[767][1860]))
