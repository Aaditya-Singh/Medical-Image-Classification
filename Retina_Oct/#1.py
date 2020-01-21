'''
import os
path = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/train/CNV/'
i = 1
for file in os.listdir(path) :
    os.rename(os.path.join(path, file), os.path.join(path, 'CNV_' + str(i) + '.jpeg'))
    i += 1

path = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/train/DME'
i = 1
for file in os.listdir(path) :
    os.rename(os.path.join(path, file), os.path.join(path, 'DME_' + str(i) + '.jpeg'))
    i += 1

path = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/train/DRUSEN'
i = 1
for file in os.listdir(path) :
    os.rename(os.path.join(path, file), os.path.join(path, 'DRUSEN_' + str(i) + '.jpeg'))
    i += 1

path = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/train/NORMAL'
i = 1
for file in os.listdir(path) :
    os.rename(os.path.join(path, file), os.path.join(path, 'NORMAL_' + str(i) + '.jpeg'))
    i += 1
'''

import os
import shutil

src = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/train/DRUSEN/'
dst = '/Users/apple/tensorflow3/AsItIs/Retina_OCT/retina_oct/train/'
for file in os.listdir(src) :
    shutil.move(src + file, dst)
