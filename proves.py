import os
from muscima.io import parse_cropobject_list
from utils import *
from convert import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Funzione che cicla su tutti i file xml
def ciclone():
    # TODO capire differenza tra i due XML
    CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_manual'
    # CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_withstaff'

    cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
    # per debuggare
    cropobject_fnames = cropobject_fnames[:5]

    docs = [parse_cropobject_list(f) for f in cropobject_fnames]

    for docID in range(len(docs)):
        doc = docs[docID]

        # prendo dall'xml l'id del writer (w) e dello spartito (p)
        w = doc[0].uid[31:33]
        p = doc[0].uid[36:38]

        imgPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/image/p0" + p + ".png"
        imgStaffPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/gt/p0" + p + ".png"

        imgStaff = mpimg.imread(imgStaffPath)

        horizontalProjection = np.sum(imgStaff, axis=1)

        pentasSeparators = getPentasSeparators(horizontalProjection)

        notesAnnotations = getOrderedNotesAnnotations(doc, pentasSeparators)

        # per plottare la proiezione orizzontale si deve prima trasformare il vettore in una lista
        # plt.plot(sums.tolist())
        # plt.show()

        imgStaff = getPreprocessedStaffImage(imgStaff)
        imgStaffLedgers = getStaffImageWithLedgers(imgStaff, doc)

        plt.imshow(imgStaff, cmap="gray")
        plt.show()

        notesPositions = getNotesPentasPositions(imgStaff, imgStaffLedgers, notesAnnotations)
        print(notesPositions)


# funzione che taglia le immagini in patches


# TODO aggiungere notazioni a patches... continuare la cosa!
# TODO funzione che prende (upper, lower) e restituisce la classe, ovvero un intero appartenente a [-5, 5]
# TODO salvare in un dizionario

# TODO riorganizzare file py e funzioni (utils.py, prova.py, train_net.py, test_net.py, convert.py (da muscima a boxes))
# TODO usare parser
if __name__ == "__main__":
    prova = 2
    if prova == 1:
        pass
        # convert("01", "10", 128, 128)
    if prova == 2:
        ciclone()
