import keras
import os
import numpy as np
from scipy.io import loadmat
import cv2
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Dropout
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add


data_processed_dir = "data-processed/"

img_width, img_height = 224, 224
num_channels = 3
batch_size = 64
num_epochs = 100000
num_classes = 196
patience = 50
verbose = 1

def createModel():
    input_tensor = Input(shape=(img_height, img_width, 3))
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    net = base_model.output
    net = GlobalAveragePooling2D()(net)
    net = Dense(1024, activation='relu')(net)
    net = Dense(num_classes, activation='softmax')(net)

    model = Model(base_model.input, net)
    for layer in base_model.layers:
        layer.trainable = True
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_processed_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_processed_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'logs/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

    model = createModel()

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=verbose)