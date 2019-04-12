import glob
import os
import numpy as np
import sys

def derange(array,maxTry=1000):
    # shuffles a iterable ensuring that none of the elements
    # remains at its original position
    c = 0
    while True:
        c += 1
        d_array = np.random.permutation(array)
        if all(array != d_array):
            break
        elif c > maxTry:
                print('Maximum number of dearangement attempts reached ('+str(maxTry)+'). Aborting...')
                break

    return d_array

mainDir = sys.argv[1]
outPairArrayPath = sys.argv[2]  ### .npy file

clases = os.listdir(mainDir)
N = len(clases)
dataPairs = list()

for n,cl1 in enumerate(clases):
    print('Progreso',n,N)
    folder = mainDir + os.sep + cl1
    filePaths = glob.glob(folder+'/*')

    otherClassPaths = list()
    for cl2 in clases:
        if cl2 != cl1:
            folder2 = mainDir + os.sep + cl2
            otherClassPaths.append(glob.glob(folder2+'/*'))
    
    filePaths = np.array(filePaths)
    otherClassPaths = np.concatenate(otherClassPaths)
    np.random.shuffle(filePaths)
    np.random.shuffle(otherClassPaths)

    minL = min(len(filePaths),len(otherClassPaths))
    filePaths = filePaths[0:minL]
    otherClassPaths = otherClassPaths[0:minL]

    l = len(filePaths)
    filePaths1 = filePaths[0:l]
    filePaths2 = derange(filePaths1)
    for i in range(l):
        dataPairs.append([filePaths1[i],filePaths2[i],1])
        dataPairs.append([filePaths1[i],otherClassPaths[i],0])
    
dataPairs = np.array(dataPairs)
np.save(outPairArrayPath,dataPairs)
print('Numero total de parejas: '+str(dataPairs.shape[0]))

    
