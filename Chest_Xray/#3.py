## Data augmentation

import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import os
folder_of_training_images = '/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/chest_xray/train/'
paths_of_training_images = os.listdir(folder_of_training_images)
while '.DS_Store' in paths_of_training_images : paths_of_training_images.remove('.DS_Store')

#folder_of_testing_images = '/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/chest_xray/Large_Test/'
#paths_of_testing_images = os.listdir(folder_of_testing_images)
#while '.DS_Store' in paths_of_testing_images : paths_of_testing_images.remove('.DS_Store')

folder_of_validation_images = '/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/chest_xray/val/'
paths_of_validation_images = os.listdir(folder_of_validation_images)
while '.DS_Store' in paths_of_validation_images : paths_of_validation_images.remove('.DS_Store')

def augment_data(folder_of_files, paths_of_files) :
    import cv2
    for file in paths_of_files :
        Image = cv2.imread(folder_of_files + file)
        Mirror_Image = cv2.flip(Image, 1)
        cv2.imwrite(folder_of_files + 'Mirror_' + file, Mirror_Image)
        #Inverted_Image = cv2.flip(Image, 0)
        #cv2.imwrite(folder_of_files + 'Inverted_' + file, Inverted_Image)

augment_data(folder_of_validation_images, paths_of_validation_images)
augment_data(folder_of_training_images, paths_of_training_images)
#augment_data(folder_of_testing_images, paths_of_testing_images)