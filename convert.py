import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils import *
from muscima.io import parse_cropobject_list
import os
import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom
from tqdm import tqdm

docDiProva = parse_cropobject_list('CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_manual/'
                                   'CVC-MUSCIMA_W-01_N-10_D-ideal.xml')
dataDir = 'MNR2019'
imagesDir = 'MNR2019/JPEGImages'
annotationsDir = 'MNR2019/Annotations'
globalPatchesCounter = 1


# Ritorna una lista di immagini contenenti solo le staffs, con i rispettivi OFFSET sulle y
# In input si ha l'immagine di partenza
def getStaffsFromImage(img, imgStaff):
    horizontalProjection = np.sum(imgStaff, axis=1)
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


# Ritorna l'immagine, quella con sole staff e le note ordinate
# In input si ha il documento XML di supervisione
def inizializeImagesAndNotes(doc):
    w = doc[0].uid[31:33]
    p = doc[0].uid[36:38]
    imgPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/image/p0" + p + ".png"
    imgStaffPath = "CVCMUSCIMA/CvcMuscima-Distortions/ideal/w-" + w + "/gt/p0" + p + ".png"

    img = mpimg.imread(imgPath)
    imgStaff = mpimg.imread(imgStaffPath)

    notes = getOrderedNotesAnnotations(doc, imgStaff)
    return img, imgStaff, notes


# Ritorna il root del file xml da salvare
# In input si hanno filename (000001, 000002..) e le dimensioni
def inizializeElementTree(filename, width, height):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "MNR2019"
    ET.SubElement(annotation, "filename").text = filename
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "The MNR2019 Database"
    ET.SubElement(source, "annotation").text = "MNR2019"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)

    return annotation


# Indenta l'xml per avere una lettura migliore
# In input si ha l'xml
def prettify(xml):
    """Return a pretty-printed XML string for the Element.
    """
    xmlString = ET.tostring(xml, 'utf-8')
    reparsed = minidom.parseString(xmlString)
    return reparsed.toprettyxml(indent="\t")

# Ritorna la classe della nota in base agli indici up e low
def getClassFromPositions(u, l):
    return l-u


# Cancella i file nelle cartelle di destinazione e crea immagini delle patch e file xml
# In input si hanno le dimensioni delle patches e il documento xml di supervisione (1 IMMAGINE)
def convert(dimX, dimY, doc):
    img, imgStaff, notes = inizializeImagesAndNotes(doc)
    global globalPatchesCounter
    pentasLimits = getPentasLimits(getHorizontalProjection(imgStaff))
    stopValue = ((pentasLimits[0][1] - pentasLimits[0][0]) / 4) * 1.5

    staffs = getStaffsFromImage(img, imgStaff)
    imgStaff = getPreprocessedStaffImage(imgStaff)
    boxes = []
    for i, staff in enumerate(staffs):
        patches = getPatchesFromStaff(staff[0], dimX, dimY)
        for patch in patches:
            filename = '{0:06d}'.format(globalPatchesCounter)
            mpimg.imsave(imagesDir + '/' + filename + '.jpg', patch[0], cmap='gray')

            xml = inizializeElementTree(filename + '.jpg', patch[0].shape[1], len(patch[0]))

            for note in notes[i]:
                # scarta gli elementi nell'xml che non si vogliono
                if not note.clsname.startswith("notehead"):
                    continue

                # scarta gli elementi che non appartengono a questa patch
                # ridimensiona le misure e aggiungi ai boxes i due punti limiti
                t = note.top
                if not staff[1][0] < t < staff[1][1]:
                    continue
                t = t - staff[1][0]
                t = getResizedMeasure(patch[1][1] - patch[1][0], dimX, t)

                l = note.left
                if not patch[1][0] < l < patch[1][1]:
                    continue
                l = l - patch[1][0]
                l = getResizedMeasure(patch[1][1] - patch[1][0], dimX, l)

                w = note.width
                w = getResizedMeasure(patch[1][1] - patch[1][0], dimX, w)
                if l + w >= len(patch[0]):
                    w = len(patch[0]) - l - 1
                if w <= 3:
                    continue
                h = note.height
                h = getResizedMeasure(patch[1][1] - patch[1][0], dimX, h)
                if t + h >= len(patch[0]):
                    h = len(patch[0]) - t - 1
                if h <= 3:
                    continue

                boxes.append([l, t, l + w, t + h])
                object = ET.SubElement(xml, "object")

                classe = None
                if isInsideStaff(note, pentasLimits):
                    up, low = getInsideStaffNotePosition(imgStaff, note, stopValue)
                    classe = getClassFromPositions(up, low)
                else:
                    classe = "OutOfStaffs"

                ET.SubElement(object, "name").text = str(classe)
                box = ET.SubElement(object, "bndbox")
                ET.SubElement(box, "xmin").text = str(l)
                ET.SubElement(box, "ymin").text = str(t)
                ET.SubElement(box, "xmax").text = str(l + w)
                ET.SubElement(box, "ymax").text = str(t + h)
            with open(annotationsDir + '/' + filename + '.xml', "w") as f:
                f.write(prettify(xml).split("\n", 1)[1])

            globalPatchesCounter = globalPatchesCounter + 1
    return boxes


# Ritorna True se le cartelle per il salvataggio dei dati esistono, False altrimenti
def checkAndClearDirectories():
    if os.path.exists(dataDir):
        if os.path.exists(annotationsDir):
            clearDirectory(annotationsDir)
            if os.path.exists(imagesDir):
                clearDirectory(imagesDir)
                print('All directories exist')
                return True
            else:
                print("I/O error (JPEGImages)")
                return False
        else:
            print("I/O error (Annotations)")
            return False
    else:
        print("I/O error (MNR2019)")
        return False


# La funzione elimina i file all'interno della cartella che ha percorso = 'directorPath'
def clearDirectory(directoryPath):
    for file in os.listdir(directoryPath):
        file_path = os.path.join(directoryPath, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def main(debug=False):
    CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_manual'
    # CROPOBJECT_DIR = 'CVCMUSCIMA/MUSCIMA++/v1.0/data/cropobjects_withstaff'

    print("Reading files from MUSCIMA...")
    print()
    cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in tqdm(os.listdir(CROPOBJECT_DIR))]
    # per debuggare
    if debug:
        cropobject_fnames = cropobject_fnames[70:71]

    print("Reading documents from files...")
    print()
    docs = [parse_cropobject_list(f) for f in tqdm(cropobject_fnames)]

    if not checkAndClearDirectories():
        return


    for docID in tqdm(range(len(docs))):
        doc = docs[docID]
        convert(128, 128, doc)


if __name__ == '__main__':
    main(False)
