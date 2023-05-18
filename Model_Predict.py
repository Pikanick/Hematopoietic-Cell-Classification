# Main Deep Learning Architecture
# name: Model_Predict.py
# author: mbwhiteh@sfu.ca
# date: 2022-04-10

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
LABEL_LIST = ['BLA', 'LYT', 'NGB', 'NGS']
# Please set the IMG PATH
IMG_PATH = "./Sample_Images/BLA_06642.jpg"

model = keras.models.load_model('./Models/Final-Model.hdf5')

# Load the image
img_in = keras.utils.load_img(IMG_PATH, target_size=(128, 128))
normalize_img = np.array([keras.utils.img_to_array(img_in)*(1/255)])
# Show Image
plt.imshow(img_in)
plt.show()
# Predict Image
results  = model.predict_step(normalize_img)
pred_class = np.argmax(results)
print(f'{LABEL_LIST}')
print(f'{results[0]}')
print(f'Predicted Class = {LABEL_LIST[pred_class]}')
print(f'True Class = {IMG_PATH[16:19]}')