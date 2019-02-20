import os
from muscima.io import parse_cropobject_list
import pprint as pp

import matplotlib

if os.environ["DISPLAY"].startswith("localhost"):  # se la GUI non è disponibile (ovvero se sono su un server)
    matplotlib.use("Agg")  # necessario per salvare immagini di matplotlib quando lo script gira su un server senza GUI
    isGuiAvailable = False
    print("GUI not found")
else:
    isGuiAvailable = True
    print("GUI found")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from conversion.utils import *
from conversion.convert_to_voc import *


# Funzione che cicla su tutti i file xml
def main():
    # TODO capire differenza tra i due XML
    CROPOBJECT_DIR = 'data/CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_manual'
    # CROPOBJECT_DIR = 'data/CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_withstaff'

    cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
    # per debuggare
    cropobject_fnames = cropobject_fnames[:5]

    docs = [parse_cropobject_list(f) for f in cropobject_fnames]

    for docID in range(len(docs)):
        doc = docs[docID]

        # prendo dall'xml l'id del writer (w) e dello spartito (p)
        w = doc[0].uid[31:33]
        p = doc[0].uid[36:38]

        imgPath = "data/CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/image/p0" + p + ".png"
        imgStaffPath = "data/CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/gt/p0" + p + ".png"

        imgStaff = mpimg.imread(imgStaffPath)
        img = mpimg.imread(imgPath)

        horizontalProjection = np.sum(imgStaff, axis=1)

        pentasSeparators = getPentasSeparators(horizontalProjection)

        notesAnnotations = getOrderedNotesAnnotations(doc, imgStaff)

        # per plottare la proiezione orizzontale si deve prima trasformare il vettore in una lista
        # plt.plot(sums.tolist())
        # plt.show()

        imgStaff = getPreprocessedStaffImage(imgStaff)
        imgStaffLedgers = getStaffImageWithLedgers(imgStaff, doc)

        notesPositions = getNotesPentasPositions(imgStaff, imgStaffLedgers, notesAnnotations)
        print(notesPositions)
        if isGuiAvailable:
            plt.imshow(img, cmap="gray")
            plt.show()

        # per ora, se sono su un server senza GUI evito di fare i plot, però potrei salvare l'immagine in questo modo:
        # else:
        #     plt.savefig("nome_immagine.png")
        #     plt.clf()


# TODO usare parser
if __name__ == "__main__":
    main()
