# Binary Classification
## Step 3 : Fine tune the randomly initialized fully connected layers

import tensorflow as tf
from keras.models import load_model, Model, Sequential
from keras.layers import Flatten, Dense, Dropout
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Input
from keras.preprocessing import image
import keras.backend as K
K.set_image_data_format('channels_last')

#reference_model = load_model('/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/daydream_model.h5')
#reference_model.summary()

## Input shape is (139,139,3) for the base model VGG19
input = Input(shape = (150,150,3))
## Reference model adds two dense layers onto base_model
from keras.regularizers import l2
base_model = InceptionV3(input_tensor = input, weights = 'imagenet', include_top = False)
reference_model = Sequential()
reference_model.add(base_model)
reference_model.add(Flatten())
reference_model.add(Dense(1336, kernel_regularizer = l2(0.01)))
reference_model.add(Dropout(0.5))
reference_model.add(Dense(1336, kernel_regularizer = l2(0.01)))
reference_model.add(Dropout(0.5))
reference_model.add(Dense(1, activation = 'sigmoid'))
reference_model.summary()

## Freeze the base_model layers before training
for layer in base_model.layers :
    layer.trainable = False


## Freeze all the layers except the second dense layer
#for layer in reference_model.layers[:2] :
   #layer.trainable = False

#for layer in reference_model.layers[5:] :
    #layer.trainable = False

## Compile the model
from keras.optimizers import SGD, Nadam
sgd = SGD(lr = 0.00001, decay = 1e-6, momentum = 0.9, nesterov = True)
nadam = Nadam(lr = 0.00002, beta_1 = 0.9, beta_2 = 0.999)
reference_model.compile(loss = 'binary_crossentropy', optimizer = nadam, metrics = ['mae','acc'])

## Define a function to convert images from image path to numpy array
def image_to_array(image_path) :
    img = image.load_img(image_path, target_size=(150, 150))
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    array = (array - np.mean(array))/np.max(array)                                  # Mean Normalization
    return array

## Define a function to return the array and label of an image from the image path
def image_array_and_label(image_path) :
    if 'Normal' in image_path :
        label = np.array([0])
    elif 'Pneumonia' in image_path :
        label = np.array([1])
    array = image_to_array(image_path)
    return [array, label]

## Define generator
import os
folder_of_training_images = '/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/chest_xray/train/'
paths_of_training_images = os.listdir(folder_of_training_images)
while '.DS_Store' in paths_of_training_images : paths_of_training_images.remove('.DS_Store')

folder_of_testing_images = '/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/chest_xray/test/'
paths_of_testing_images = os.listdir(folder_of_testing_images)
while '.DS_Store' in paths_of_testing_images : paths_of_testing_images.remove('.DS_Store')

folder_of_validation_images = '/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/chest_xray/val/'
paths_of_validation_images = os.listdir(folder_of_validation_images)
while '.DS_Store' in paths_of_validation_images : paths_of_validation_images.remove('.DS_Store')

def generator(folder_of_files, paths_of_files) :
    for file in paths_of_files :
        yield(image_array_and_label(folder_of_files + file))

train_generator = generator(folder_of_training_images, paths_of_training_images)
test_generator = generator(folder_of_testing_images, paths_of_testing_images)
val_generator = generator(folder_of_validation_images, paths_of_validation_images)

## Train the model using model.fit_generator
reference_model.fit_generator(train_generator, epochs = 1, steps_per_epoch = 5724, verbose = 1, validation_data = val_generator, validation_steps = 16)
reference_model.save('/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/daydream_model.h5')
[loss, mean_absolute_error, accuracy] = reference_model.evaluate_generator(test_generator, steps = 624)