import tensorflow as tf
import os
import numpy as np

def scriptPath():
    return os.path.realpath(__file__)


def convUnit(layer,filters,dropout=0):

    layer = tf.keras.layers.Conv2D(filters,kernel_size=3,padding='same',activation='relu')(layer)
   # layer = tf.keras.layers.BatchNormalization()(layer)
    # layer = tf.keras.layers.Activation('relu')(layer)
    if dropout:
         layer = tf.keras.layers.Dropout(rate=dropout)(layer)
    return layer

    
def Network(inputs):

    conv1 = convUnit(inputs,32)
    mp1 = tf.keras.layers.MaxPooling2D(pool_size = 2,padding = 'same')(conv1)

    conv2 = convUnit(mp1,32)
    mp2 = tf.keras.layers.MaxPooling2D(pool_size = 2,padding = 'same')(conv2)

    conv3 = convUnit(mp2,32)

    flat = tf.keras.layers.Flatten()(conv3)
    dense = tf.keras.layers.Dense(units = 128,activation = 'sigmoid')(flat)

    return dense