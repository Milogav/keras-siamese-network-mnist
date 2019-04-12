import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
import tensorflow.keras.backend as K
from padNet import Network,scriptPath
import numpy as np
from utils import *
from time import time,sleep
import numpy as np
import glob
import cv2
from shutil import copyfile
import subprocess
from batchGenerator import batchGenerator
  
# def contrastive_loss(label, distance):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 0.7
#     square_pred = K.square(distance)
#     margin_square = K.square(K.maximum(margin - distance,0))
#     return K.mean(label * square_pred + (1 - label) * margin_square)

def contrastive_loss(Y_true, D):
    margin = 0.7
    return K.mean(Y_true*K.square(D)+(1 - Y_true)*K.maximum((margin-D),0))

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square,K.epsilon()))

def cosine_distance(vects):
    return 1 - euclidean_distance(vects) / 2

def abs_diff(vects):
    x, y = vects
    return K.abs(x - y)

def accuracy(y_true,distances):
    dist_thr = 0.5  ### fixed distance threshold between classes
    return K.mean(K.equal(y_true, K.cast(distances < dist_thr, y_true.dtype)))

class customCallback(tf.keras.callbacks.Callback):

    def __init__(self,cfg,cfgPath):
        self.cfg = cfg
        self.cfgPath = cfgPath

    def on_epoch_begin(self,epoch, logs=None):
        lr_val = K.eval(self.model.optimizer.lr)
        print('\nLearning rate: '+str(lr_val))
    
    def on_epoch_end(self,epoch, logs=None):
        self.cfg['last_epoch'] = epoch
        saveCfgFile(self.cfg,self.cfgPath)
    

################################ LOAD TRAIN CONFIG FILE
cwd = os.path.dirname(os.path.realpath(__file__))
cfgPath = cwd+'/cfg.json'
cfg = loadCfgFile(cfgPath)

################################ SET UP LEARNING RATE UPDATING
def lrUpdatePolicy(epoch):
    init_lr = cfg['init_learning_rate']
    red_factor = 0.8
    return init_lr * (red_factor ** epoch)

################################ DEFINE CALLBACKS
trainMode = cfg['train_mode']

currTime = getCurrentTime()
summariesPath = cfg['output_dir'] + os.sep + 'tensorboard' + os.sep + currTime

tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir = summariesPath) #### log each 200 samples
lrSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(lrUpdatePolicy)

bestModelPath = cfg['output_dir'] + os.sep + 'bestModel.h5'
bestModelSaverCallback = tf.keras.callbacks.ModelCheckpoint(bestModelPath,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='auto',
                                                    period=1)

lastModelPath = cfg['output_dir'] + os.sep + 'lastModel.h5'
lastModelSaverCallback = tf.keras.callbacks.ModelCheckpoint(lastModelPath,
                                                    verbose=0,
                                                    save_best_only=False,
                                                    save_weights_only=False,
                                                    mode='auto',
                                                    period=1)

trainLogPath = cfg['output_dir'] + os.sep + 'training.log'
if trainMode == 'start':
    appendLog = False
else:
    appendLog = True
trainLogCallback = tf.keras.callbacks.CSVLogger(trainLogPath,append = appendLog)
stopNaNCallback = tf.keras.callbacks.TerminateOnNaN()
myCallback = customCallback(cfg,cfgPath)

############################## MODEL INITIALIZATION OR LOADING

if trainMode == 'start':
    print('Starting training...')
    init_epoch = 0
    train_step = 0

    ######## define base model for feature extraction
    net_in = tf.keras.Input(cfg['input_shape'])
    net_out = Network(net_in)

    feature_extractor = tf.keras.Model(inputs = net_in,outputs = net_out)

    ######## apply the feature extraction model in a siamese fashion and compile the resulting siamese model operation
    x1 = tf.keras.Input(cfg['input_shape'])
    x2 = tf.keras.Input(cfg['input_shape'])

    emb1 = feature_extractor(x1)
    emb2 = feature_extractor(x2)

    dist = tf.keras.layers.Lambda(euclidean_distance)([emb1,emb2])
    # abs_diff_layer = tf.keras.layers.Lambda(lambda tensors : tf.abs(tensors[0] - tensors[1]))
    # diff = abs_diff_layer((v1,v2))
    # logits = tf.keras.layers.Dense(units = 1,activation = 'sigmoid')(diff)
    model = tf.keras.Model(inputs = [x1,x2],outputs = dist)
    model.compile(optimizer = 'adam',
                  loss = contrastive_loss,
                  metrics = [accuracy])

elif trainMode == 'resume':
    lastModelPath = cfg['output_dir'] + os.sep + 'lastModel.hdf5'
    model = tf.keras.models.load_model(lastModelPath)
    init_epoch = model.last_epoch + 1
    train_step = 0
    
    print('\n\nResuming training from file: '+lastModelPath+' and epoch: '+str(init_epoch -1))
    
    if init_epoch >= cfg['train_epochs']:
        print('The target number of epochs specified ('+str(cfg['train_epochs'])+') has already been reached. To continue training, increase the number of epochs in the cfg json file')
        exit()

######################## TRAINING AND VALIDATION DATA GENERATORS CREATION
train_pair_array = np.load(cfg['train_data_pairs'])
val_pair_array = np.load(cfg['val_data_pairs'])

train_generator = batchGenerator(train_pair_array,cfg['batch_size'],cfg['input_shape'])
val_generator = batchGenerator(val_pair_array,cfg['batch_size'],cfg['input_shape'])

##### MODEL FITTING ###### 
# subprocess.Popen(["tensorboard", "--port","3468","--logdir",summariesPath])
# subprocess.Popen(["google-chrome","http://localhost:3468"])

print('\n\nTensorboard command:\ntensorboard --port 3468 --logdir='+summariesPath+'\n\n')

model.fit_generator(train_generator.generator(),
                    steps_per_epoch = train_generator.steps_per_epoch,
                    initial_epoch = init_epoch,
                    epochs = cfg['train_epochs'],
                    validation_data = val_generator.generator(),
                    validation_steps = val_generator.steps_per_epoch,
                    callbacks = [tensorboardCallback,
                                lrSchedulerCallback,
                                bestModelSaverCallback,
                                lastModelSaverCallback,
                                trainLogCallback,
                                stopNaNCallback,
                                myCallback])
