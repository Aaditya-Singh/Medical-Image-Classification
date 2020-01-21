# Binary or Multi-Class Classification
## Step_1 : Training the fully connected layers

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Dropout
import numpy as np
from keras.applications.vgg19 import preprocess_input
from keras.layers import Input
from keras.preprocessing import image
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.regularizers import l2

## Construct a model with 3 Dense and one softmax layers
model_layers_input = Input(shape = (139,139,3))
model_layers_output = Flatten()(model_layers_input)
model_layers_output = Dense(8192, kernel_regularizer= l2(0.01))(model_layers_output)   # 4*4*512 = 8192
model_layers_output = Dropout(0.5)(model_layers_output)                                 # Dropout
model_layers_output = Dense(1336, kernel_regularizer= l2(0.01))(model_layers_output)
model_layers_output = Dropout(0.5)(model_layers_output)                                 # Dropout
model_layers_output = Dense(1336, kernel_regularizer= l2(0.01))(model_layers_output)
model_layers_output = Dropout(0.5)(model_layers_output)                                 # Dropout
model_layers_output = Dense(3, activation = 'softmax')(model_layers_output)
model_layers = Model(model_layers_input, model_layers_output)
model_layers.summary()

# Freeze the input layers
for layer in model_layers.layers[:3] :
    layer.trainable = False

## Compile the model
from keras.optimizers import SGD,Nadam
sgd = SGD(lr = 0.00001, decay = 1e-6, momentum = 0.9, nesterov = True)
nadam = Nadam(lr = 0.00002, beta_1 = 0.9, beta_2 = 0.999)
model_layers.compile(loss = 'categorical_crossentropy', optimizer = nadam, metrics = ['mae', 'acc'])

## Define a function to convert images from image path to numpy array
def image_to_array(image_path) :
    img = image.load_img(image_path, target_size=(139, 139))
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    array = (array - np.mean(array))/np.max(array)                                  ## Mean Normalization
    return array

## Define a function to return the array and label of an image from the image path
def Binary_Label(image_path) :
    import numpy as np
    if 'Normal' in image_path : return np.array([0])
    else : return np.array([1])

def Category_Label(image_path) :
    from keras.utils import to_categorical
    import numpy as np
    if 'Normal' in image_path :
        label = np.array([0])
    elif 'Bacteria' in image_path :
        label = np.array([1])
    elif 'Virus' in image_path :
        label = np.array([2])
    one_hot_label = to_categorical(label, num_classes = 3)
    return one_hot_label

def image_array_and_label(image_path) :
    array = image_to_array(image_path)
    label = Category_Label(image_path)
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

model_layers.fit_generator(train_generator, epochs = 1, steps_per_epoch = 11448, verbose = 1, validation_data = val_generator, validation_steps = 32)

model_layers.save('/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/model_layers_category.h5')
model_layers.save_weights('/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/model_layers_weights_category.h5')
