

"""
This program provides all the different models used in the training purposes
"""

# Importing all the necessary libraies and functions
import numpy as np
import pandas as pd
import keras
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2


def mobilenet(num_classes, input_shape = (224,224,3)):
    """
    This function creates the mobilenet model based on the transfer learning
    concepts

    input_shape: input shape of the image
    num_classes: number of classes in the output

    return: returns the mobile net model
    """
    img_input = Input(input_shape)

    model = keras.applications.mobilenet.MobileNet(include_top=False, weights= 'imagenet', input_shape= input_shape)
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = Model(model.input, output)
    return model

if __name__ == "__main__":
    '''
    #mobilenet(num_classes = 2)
    #mini_xception((48, 48, 3), 5, regularization = l2(0.01))
    '''
