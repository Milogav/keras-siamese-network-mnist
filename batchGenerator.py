import glob
from natsort import natsorted
import numpy as np
import os
from utils import derange,batchSplit,normalize,normalizeRGB
import cv2
from time import sleep,time

def procData(img,input_shape):
    # img = cv2.resize(img,(input_shape[0],input_shape[1])).astype(float)
    img = img.astype(np.float32) / 255
    # img = np.expand_dims(normalize(img),axis=2)
    img = np.expand_dims(img,axis=2)
    return img


class batchGenerator():

    def __init__(self, pair_path_array, batch_size, input_shape ,maxPairsPerEpoch = np.inf):

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.pair_path_array = pair_path_array
        self.maxPairsPerEpoch = maxPairsPerEpoch
        self.input_shape = tuple(input_shape[0:2])
        self.numPairs = min(maxPairsPerEpoch,self.pair_path_array.shape[0])
        self.steps_per_epoch = self.numPairs // self.batch_size 
        
    def generator(self):
        
        while True:
            np.random.shuffle(self.pair_path_array)
            path_array = self.pair_path_array[0:self.numPairs,:]
            batches = batchSplit(path_array,self.batch_size)

            for B in batches:
                x1data = list()
                x2data = list()
                ydata = list()
 
                for pair in B:

                    x1 = cv2.imread(pair[0],0)
                    x2 = cv2.imread(pair[1],0)
                    if x1 is None or x2 is None:
                        continue

                    x1data.append(procData(x1,self.input_shape))
                    x2data.append(procData(x2,self.input_shape))
                    ydata.append([int(pair[2])])
                    
                x1data = np.array(x1data)
                x2data = np.array(x2data)
                ydata = np.array(ydata)

                yield [x1data,x2data],ydata
                
