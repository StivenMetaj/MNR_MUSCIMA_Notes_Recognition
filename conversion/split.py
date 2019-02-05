import os
import numpy as np
from xml.etree import cElementTree as ET

# GLOBAL VARIABLES
splitDir = "MNR2019/ImageSets"
layoutDir = splitDir + "/Layout"
mainDir = splitDir + "/Main"
splitDirectoriesList = [layoutDir, mainDir]
annotationsDir = "MNR2019/Annotations"

# TRAIN, VALIDATION, TEST
splitPoints = [60, 20, 20]
assert (sum(i for i in splitPoints) == 100)


def inizializePermutation(seed=0, N=21244):
    ids = np.arange(N + 1)
    ids = ids[1:]

    np.random.seed(seed)
    np.random.shuffle(ids)

    return ids


# Ritorna True se le cartelle per il salvataggio dei dati esistono, False altrimenti
def checkAndClearDirectories(directoriesList, askForClearing=True):
    # se almeno una delle due cartelle di output esiste, chiedo conferma per sovrascrivere il contenuto
    if askForClearing:
        for dir in directoriesList:
            if os.path.exists(dir):
                response = ""
                while response != 'y' and response != 'n':
                    print(dir + " it's not empty: do you want to clear it and overwrite the content? (y/n)")
                    response = input().lower()
                if response == 'n':
                    return False

    # creo cartelle o elimino contenuto
    for dir in directoriesList:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            clearDirectory(dir)

    return True


# La funzione elimina i file all'interno della cartella che ha percorso = 'directorPath'
def clearDirectory(directoryPath):
    for file in os.listdir(directoryPath):
        file_path = os.path.join(directoryPath, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def split(permutation):
    checkAndClearDirectories(splitDirectoriesList, False)
    trainEnd = int((len(permutation) / 100) * splitPoints[0])
    valEnd = trainEnd + int((len(permutation) / 100) * splitPoints[1])

    with open(layoutDir + "/trainval.txt", "w") as trainvalFile:
        with open(layoutDir + "/train.txt", "w") as trainFile:
            for i in permutation[0:trainEnd]:
                name = '{0:06d}'.format(i)
                trainFile.write(name + '\n')
                trainvalFile.write(name + '\n')

                xmlFile = annotationsDir + "/" + name + ".xml"
                xml = ET.parse(xmlFile)
                for j in xml.iter('name'):
                    cls = j.text
                    with open(mainDir + "/" + cls + "_train.txt", "a") as clsFile:
                        clsFile.write(name + '\n')
                    with open(mainDir + "/" + cls + "_trainval.txt", "a") as clsFile:
                        clsFile.write(name + '\n')

        with open(layoutDir + "/val.txt", "w") as valFile:
            for i in permutation[trainEnd + 1:valEnd]:
                name = '{0:06d}'.format(i)
                valFile.write(name + '\n')
                trainvalFile.write(name + '\n')

                xmlFile = annotationsDir + "/" + name + ".xml"
                xml = ET.parse(xmlFile)
                for j in xml.iter('name'):
                    cls = j.text
                    with open(mainDir + "/" + cls + "_val.txt", "a") as clsFile:
                        clsFile.write(name + '\n')
                    with open(mainDir + "/" + cls + "_trainval.txt", "a") as clsFile:
                        clsFile.write(name + '\n')

    with open(layoutDir + "/test.txt", "w") as testFile:
        for i in permutation[valEnd:]:
            name = '{0:06d}'.format(i)
            testFile.write(name + '\n')

            xmlFile = annotationsDir + "/" + name + ".xml"
            xml = ET.parse(xmlFile)
            for j in xml.iter('name'):
                cls = j.text
                with open(mainDir + "/" + cls + "_test.txt", "a") as clsFile:
                    clsFile.write(name + '\n')


if __name__ == '__main__':
    permutation = inizializePermutation()
    split(permutation)
