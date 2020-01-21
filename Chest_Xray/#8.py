# Image label
def Binary_Label(image_path) :
    import numpy as np
    if 'Normal' in image_path : return np.array([0])
    else : return np.array([1])

print(Binary_Label('/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/chest_xray/val/Pneumonia_Bacteria_Val_2.jpeg'))

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

print(Category_Label('/Users/apple/tensorflow3/AsItIs/Chest_X_Ray/chest_xray/val/Pneumonia_Bacteria_Val_2.jpeg'))
