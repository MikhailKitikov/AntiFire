from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile
from keras.applications import ResNet50V2
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Input, Conv2D
from keras.models import Model
from collections import deque
import tensorflow as tf
import numpy as np


def create_model_head(baseModel):
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)

    new_model = Model(inputs=baseModel.input, outputs=headModel)
    return new_model


def load_mobilenetv2():
    weights_path = '../Models/Trained models/mobileNetv2.h5'
    baseNet = MobileNetV2(weights=None, include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model = create_model_head(baseNet)
    model.load_weights(weights_path)
    return model


def load_nasnetmobile():
    weights_path = '../Models/Trained models/nasnetMobile.h5'
    baseNet = NASNetMobile(weights=None, include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model = create_model_head(baseNet)
    model.load_weights(weights_path)
    return model


def load_resnet50():
    weights_path = '../Models/Trained models/resnet50v2.h5'
    baseNet = ResNet50V2(weights=None, include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model = create_model_head(baseNet)
    model.load_weights(weights_path)
    return model


def load_FireNet():
    model = Sequential()
    data_input_shape = (224,224,3)

    model.add(Convolution2D(128, (3,3),padding='same',activation='relu', input_shape=data_input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', name='high_output'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    weights_path = '../Models/Trained models/FireNet_large_new.h5'
    model.load_weights(weights_path)
    return model


def load_FireNetStack():
    model = load_FireNet()
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('high_output').output)
    
    return intermediate_layer_model


def load_FireNetMobile():
    model = Sequential()
    data_input_shape = (64,64,3)

    model.add(Convolution2D(64, (3,3),padding='same',activation='relu', input_shape=data_input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(32, (5,5),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(32, (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', name='low_output'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    weights_path = '../Models/Trained models/FireNetMobile.h5'
    model.load_weights(weights_path)
    return model


def load_FireNetMobileStack():
    model = load_FireNetMobile()
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('low_output').output)
    
    return intermediate_layer_model


def load_LSTM():
    n_timesteps = 10
    n_features = 640

    model2 = Sequential()
    model2.add(LSTM(100, input_shape=(n_timesteps, n_features), return_sequences=True))
    model2.add(Dropout(0.5))
    model2.add(LSTM(200, return_sequences=False))
    model2.add(Dropout(0.5))
    model2.add(Dense(100, activation='relu'))
    model2.add(Dense(1, activation='sigmoid'))

    weights_path = '../Models/Trained models/LSTM.h5'
    model2.load_weights(weights_path)
    
    return model2

