# Multi-Class Classification
## Step_2 : Fine tune the trained fully connected layers

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense
import numpy as np
from keras.applications import vgg16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.layers import Input
from keras.preprocessing import image
import keras.backend as K
K.set_image_data_format('channels_last')

## Input shape is (139,139,3) for the base model VGG16
input = Input(shape = (139,139,3))
base_model = VGG19(input_tensor = input, weights = 'imagenet', include_top = False)

## Import the model named model_layers from the h5 file 'model_layers.h5'
from keras.models import load_model
model_layers = load_model('/Users/apple/tensorflow3/AsItIs/Retina_OCT/model_layers_category.h5')

## Create a new model which takes the last two dense layers of model_layers and adds them onto the base_model
model = Sequential()
model.add(base_model)
model.add(Flatten())                                            # output shape is (?,8192)
model.add(model_layers.layers[4])                               # dense layer which outputs (?,1336)
model.add(model_layers.layers[5])                               # dropout layer
model.add(model_layers.layers[6])                               # dense layer which outputs (?,1336)
model.add(model_layers.layers[7])                               # dropout layer
model.add(model_layers.layers[8])                               # softmax layer which outputs (?,4)
# Load weigths into the layers of model_layers
model.load_weights('/Users/apple/tensorflow3/AsItIs/Retina_OCT/model_layers_weights_category.h5', by_name = True)
model.summary()

## Freeze the base_model layers before training
for layer in base_model.layers :
    layer.trainable = False

## Compile the model
from keras.optimizers import SGD, Nadam
sgd = SGD(lr = 0.00001, decay = 1e-6, momentum = 0.9, nesterov = True)
nadam = Nadam(lr = 0.00002, beta_1 = 0.9, beta_2 = 0.999)
model.compile(loss = 'categorical_crossentropy', optimizer = nadam, metrics = ['mae','acc'])

## Define a function to convert images from image path to numpy array
def image_to_array(image_path) :
    img = image.load_img(image_path, target_size=(139, 139))
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    array = (array - np.mean(array))/np.max(array)                                  # Mean Normalization
    return array

## Define a function to return the array and label of an image from the image path
def image_array_and_label(image_path) :
    from keras.utils import to_categorical
    if 'NORMAL' in image_path :
        label = np.array([0])
    elif 'CNV' in image_path :
        label = np.array([1])
    elif 'DME' in image_path :
        label = np.array([2])
    elif 'DRUSEN' in image_path :
        label = np.array([3])
    one_hot_label = to_categorical(label, num_classes = 4)
    array = image_to_array(image_path)
    return [array, one_hot_label]

## Define generator
import os
folder_of_training_images = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/train/'
paths_of_training_images = os.listdir(folder_of_training_images)
while '.DS_Store' in paths_of_training_images : paths_of_training_images.remove('.DS_Store')

folder_of_testing_images = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/test/'
paths_of_testing_images = os.listdir(folder_of_testing_images)
while '.DS_Store' in paths_of_testing_images : paths_of_testing_images.remove('.DS_Store')

folder_of_validation_images = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/val/'
paths_of_validation_images = os.listdir(folder_of_validation_images)
while '.DS_Store' in paths_of_validation_images : paths_of_validation_images.remove('.DS_Store')

def generator(folder_of_files, paths_of_files) :
    for file in paths_of_files :
        yield(image_array_and_label(folder_of_files + file))

train_generator = generator(folder_of_training_images, paths_of_training_images)
test_generator = generator(folder_of_testing_images, paths_of_testing_images)
val_generator = generator(folder_of_validation_images, paths_of_validation_images)

## Train the model using model.fit_generator
model.fit_generator(train_generator, epochs = 1, steps_per_epoch = 25000, verbose = 1, validation_data = val_generator, validation_steps = 32)
model.save('/Users/apple/tensorflow3/AsItIs/Retina_OCT/model.h5')
[loss, mean_absolute_error, accuracy] = model.evaluate_generator(test_generator, steps = 968)






