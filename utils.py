import numpy as np
import os
from natsort import natsorted
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import json
import cv2

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

def batchSplit(dataList,batchSize):
    #### splits a data list in batches of size = batchSize
    lData = len(dataList)
    dataBatches = np.array_split(dataList,np.arange(batchSize,lData,batchSize))
    return dataBatches

def log(logfile,string,printStr = True):
    logfile.write(string+'\n')
    logfile.flush()
    if printStr:
        print(string)
    
def getCurrentTime():
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def filelist(folder,nameFilter=None):
    files = list()
    if not nameFilter:
        files = natsorted([folder+'/'+x for x in os.listdir(folder)])
    else:    
        for x in os.listdir(folder):
            if nameFilter in x:
                files.append(folder+'/'+x)
        files = natsorted(files)

    return files 

def loadCfgFile(cfgPath):
    with open(cfgPath,'r') as fp:
        cfgDict = json.load(fp)
    return cfgDict

def saveCfgFile(cfgDict,cfgPath):
    with open(cfgPath,'w') as fp:
        json.dump(cfgDict,fp,indent=4)

def normalize(img):
    img = (img - img.min()) / (img.max()-img.min())
    return img

def normalizeRGB(img):
    for i in range(3):
        img[:,:,i] = (img[:,:,i] - img[:,:,i].min()) / (img[:,:,i].max()-img[:,:,i].min())
    return img

def bgr2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def rgb2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def bgr2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


