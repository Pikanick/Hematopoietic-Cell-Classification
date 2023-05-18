# Main Deep Learning Architecture
# name: Model_Test.py
# author: mbwhiteh@sfu.ca
# date: 2022-04-10

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, GaussianNoise, MaxPooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy, Precision, Recall

from Utility_Functions import data_helper as DataUtil
import random

OUTPUT_CLASSES = 4

# High Level Parameters
batch_size = 32
lr = 0.0001
beta_1 = 0.9
epochs = 75
train_valid_split = 0.2

with open("./test_filenames.txt", 'r+') as test_fd:
    test_fnames = [f_name.strip('\n') for f_name in test_fd.readlines()]
    random.shuffle(test_fnames)
# format test labels
test_labels = [DataUtil.LABEL_DICT[f_name.split('/')[0]] for f_name in test_fnames]

initializer = keras.initializers.HeNormal()

# testing data generator
test_datagen = DataUtil.DataGenerator(test_fnames, test_labels, batch_size, OUTPUT_CLASSES)

# Start of model definition
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=(128, 128, 3), kernel_initializer= initializer, kernel_regularizer='l2'))
# L1
model.add(GaussianNoise(1))
model.add(Conv2D(256, (3, 3), kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L2 Conv
model.add(Conv2D(256, (3, 3), kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=2))
# L3 Conv
model.add(GaussianNoise(1))
model.add(Conv2D(128, (3, 3), kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L4 Conv
model.add(Conv2D(128, (3, 3), kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L5 Conv
model.add(Conv2D(128, (3, 3), kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=2))
# L6 Conv
model.add(GaussianNoise(1))
model.add(Conv2D(64, (3, 3), kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L7 Conv
model.add(Conv2D(64, (3, 3), kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Flatten())
# L8 Dense
model.add(GaussianNoise(1))
model.add(Dense(256, kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L9 Dense
model.add(Dense(128, kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.50))
# L10
model.add(Dense(64, kernel_initializer= initializer, kernel_regularizer='l2'))
model.add(BatchNormalization())
# L11 Output
model.add(Dense(OUTPUT_CLASSES))
model.add(Activation('softmax'))

# optimizer configuration
opt = keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=beta_1
)
model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=[
                  CategoricalCrossentropy(),
                  CategoricalAccuracy(),
                  Recall(thresholds=0.50, class_id=0, name='bla-recall'),
                  Recall(thresholds=0.50, class_id=1, name='lyt-recall'),
                  Recall(thresholds=0.50, class_id=2, name='ngb-recall'),
                  Recall(thresholds=0.50, class_id=3, name='ngs-recall'),
                  Precision(thresholds=0.50, class_id=0, name='bla-precision'),
                  Precision(thresholds=0.50, class_id=1, name='lyt-precision'),
                  Precision(thresholds=0.50, class_id=2, name='ngb-precision'),
                  Precision(thresholds=0.50, class_id=3, name='ngs-precision')
])

model.load_weights('./Weights/MV3_Weights.hdf5')
model.save('./Models/Final-Model.hdf5')
model.evaluate(test_datagen)