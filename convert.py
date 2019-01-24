import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils import *
from muscima.io import parse_cropobject_list

docDiProva = parse_cropobject_list('CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_manual/'
                                   'CVC-MUSCIMA_W-01_N-10_D-ideal.xml')


# Ritorna una lista di immagini contenenti solo le staffs, con i rispettivi OFFSET sulle y
# In input si ha l'immagine di partenza
def getStaffsFromImage(img):
    horizontalProjection = np.sum(img, axis=1)
    pentasSeparators = getPentasSeparators(horizontalProjection)
    staffsAndOffsetsY = []

    # per separatori - 1 volte taglio l'immagine di partenza e la salvo in staff
    for i in range(len(pentasSeparators) - 1):
        start, end = pentasSeparators[i], pentasSeparators[i + 1]
        staff = img[start: end]
        staffsAndOffsetsY.append((staff, (start, end)))

    return staffsAndOffsetsY


# Ritorna una lista di immagini contenenti solo le patch per il dataset
# In input si hanno l'immagine del pentagramma singolo e le dimensioni w, h
def getPatchesFromStaff(staff, w, h):
    patchesAndOffsetsX = []
    l = len(staff)

    transpStaff = staff.transpose()
    # numero di patch sequenziali
    n = int(staff.shape[1] / l)
    for i in range(n):
        # taglio il pentagramma in modo sequenziale
        start, end = i * l, (i + 1) * l
        patch = transpStaff[start: end].transpose()
        # print(str(i*l))
        # plt.imshow(patch, cmap="gray")
        # plt.show()
        patch = resizePatch(patch, w, h)
        patchesAndOffsetsX.append((patch, (start, end)))

        # creo patch casuali di ugual numero a quelle sequenziali
        start = random.randint(0, staff.shape[1] - l)
        end = start + l
        patch = transpStaff[start: end].transpose()
        patch = resizePatch(patch, w, h)
        patchesAndOffsetsX.append((patch, (start, end)))

    return patchesAndOffsetsX


# Ritorna il resize dell'immagine 'patch' secondo le dimensioni w h
def resizePatch(patch, w, h):
    return cv2.resize(patch, dsize=(w, h), interpolation=cv2.INTER_AREA)


# Ritorna la misura dopo aver fatto un resize da startDim a finalDim
# Si e' pensato al fatto che il resize è operazione lineare e quindi il calcolo deriva dalla
# proporzione [start : final = measure : X]
# In input si hanno quindi i 3 valori
def getResizedMeasure(startDimension, finalDimension, measure):
    return int((finalDimension * measure) / startDimension)


def convert(dimX, dimY, doc):
    w = doc[0].uid[31:33]
    p = doc[0].uid[36:38]
    imgPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/image/p0" + p + ".png"
    imgStaffPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/gt/p0" + p + ".png"

    img = mpimg.imread(imgPath)

    staffs = getStaffsFromImage(img)

    for staff in staffs:
        patches = getPatchesFromStaff(staff[0], dimX, dimY)
        for patch in patches:
            boxes = []
            # TODO aggiungere x1y1 x2y2 ai boxes
            for k in range(len(doc)):
                # scarta gli elementi nell'xml che non si vogliono
                if not doc[k].clsname.startswith("notehead"):
                    continue

                t = doc[k].top
                if not staff[1][0] < t < staff[1][1]:
                    continue
                t = t - staff[1][0]
                t = getResizedMeasure(patch[1][1] - patch[1][0], dimX, t)

                l = doc[k].left
                if not patch[1][0] < l < patch[1][1]:
                    continue
                l = l - patch[1][0]
                l = getResizedMeasure(patch[1][1] - patch[1][0], dimX, l)

                w = doc[k].width
                w = getResizedMeasure(patch[1][1] - patch[1][0], dimX, w)
                if l + w >= len(patch[0]):
                    w = len(patch[0]) - l - 1
                h = doc[k].height
                h = getResizedMeasure(patch[1][1] - patch[1][0], dimX, h)
                if t + h >= len(patch[0]):
                    h = len(patch[0]) - t - 1

                squaresColor = 0.8

                for i in range(w):
                    patch[0][t][l + i] = squaresColor
                    patch[0][t + h][l + i] = squaresColor

                for i in range(h):
                    patch[0][t + i][l] = squaresColor
                    patch[0][t + i][l + w] = squaresColor
            plt.imshow(patch[0], cmap="gray")
            plt.show()


if __name__ == '__main__':
    convert(128, 128, docDiProva)
