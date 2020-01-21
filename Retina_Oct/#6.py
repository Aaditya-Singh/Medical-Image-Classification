## Improve test accuracy
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np

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
folder_of_testing_images = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/custom_test/'
paths_of_testing_images = os.listdir(folder_of_testing_images)
while '.DS_Store' in paths_of_testing_images : paths_of_testing_images.remove('.DS_Store')

def generator(folder_of_files, paths_of_files) :
    for file in paths_of_files :
        yield(image_array_and_label(folder_of_files + file))

test_generator = generator(folder_of_testing_images, paths_of_testing_images)

from keras.models import load_model
reference_model = load_model('/Users/apple/tensorflow3/AsItIs/Retina_OCT/reference_model.h5')
reverie_model = load_model('/Users/apple/tensorflow3/AsItIs/Retina_OCT/reverie_model.h5')

[reference_loss, reference_mae, reference_accuracy] = reference_model.evaluate_generator(test_generator, steps = 916)
#[reverie_loss, reverie__mae, reverie__accuracy] = reverie_model.evaluate_generator(test_generator, steps = 916)