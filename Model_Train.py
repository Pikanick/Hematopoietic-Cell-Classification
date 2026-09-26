# Main Deep Learning Architecture
# name: Model_Train.py
# author: mbwhiteh@sfu.ca
# date: 2022-02-22

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, GaussianNoise, MaxPooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy, Precision, Recall

from Utility_Functions import data_helper as DataUtil
import random
import math
import os
import re

# --- Colab / resumable-training settings (overridable via environment) ---
if os.environ.get('MV3_MIXED', '1') == '1':
    # float16 compute on tensor cores: much faster on a T4, weights stay float32
    keras.mixed_precision.set_global_policy('mixed_float16')
BACKUP_DIR = os.environ.get('MV3_BACKUP', './backup')   # full training state, resumed automatically
LOG_FP = os.path.abspath(os.environ.get('MV3_LOG', './MV3-Final.csv'))

OUTPUT_CLASSES = 4
# divide conv widths (1 = paper's MV3; 2 = half-width, ~4x faster)
W = int(os.environ.get('MV3_WIDTH_DIV', 1))
NOISE = float(os.environ.get('MV3_NOISE', 1))    # GaussianNoise stddev (paper: 1)
L2 = float(os.environ.get('MV3_L2', 0.01))       # kernel L2 strength (paper/keras 'l2' default: 0.01)
MODEL_CHECKPOINTS_FP = os.path.abspath(os.environ.get('MV3_CKPT', './Weights/MV3_Weights.weights.h5'))

# High Level Parameters
batch_size = 32
lr = 0.0001
beta_1 = 0.9
epochs = int(os.environ.get('MV3_EPOCHS', 75))
train_valid_split = 0.2

# set the seed
random.seed(11)

# read training filenames from text file
with open("./train_filenames.txt", 'r+') as train_fd:
    train_fnames = [f_name.strip('\n') for f_name in train_fd.readlines()]
    # shuffle the training data
    random.shuffle(train_fnames)
# format train labels
train_labels = [DataUtil.LABEL_DICT[f_name.split('/')[0]] for f_name in train_fnames]

with open("./test_filenames.txt", 'r+') as test_fd:
    test_fnames = [f_name.strip('\n') for f_name in test_fd.readlines()]
    random.shuffle(test_fnames)
# format test labels
test_labels = [DataUtil.LABEL_DICT[f_name.split('/')[0]] for f_name in test_fnames]

initializer = keras.initializers.HeNormal()

# number of samples for training and validation
train_n = len(train_fnames)
valid_n = math.ceil(train_valid_split*train_n)

# training data generator
train_datagen = DataUtil.DataGenerator(train_fnames[valid_n:train_n], train_labels[valid_n:train_n], batch_size, OUTPUT_CLASSES)
# validation data generator
validation_datagen = DataUtil.DataGenerator(train_fnames[0:valid_n], train_labels[0:valid_n], batch_size, OUTPUT_CLASSES)
# testing data generator
test_datagen = DataUtil.DataGenerator(test_fnames, test_labels, batch_size, OUTPUT_CLASSES)

# Start of model definition
model = Sequential()

model.add(Conv2D(256 // W, (3, 3), input_shape=(128, 128, 3), kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
# L1
model.add(GaussianNoise(NOISE))
model.add(Conv2D(256 // W, (3, 3), kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L2 Conv
model.add(Conv2D(256 // W, (3, 3), kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=2))
# L3 Conv
model.add(GaussianNoise(NOISE))
model.add(Conv2D(128 // W, (3, 3), kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L4 Conv
model.add(Conv2D(128 // W, (3, 3), kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L5 Conv
model.add(Conv2D(128 // W, (3, 3), kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=2))
# L6 Conv
model.add(GaussianNoise(NOISE))
model.add(Conv2D(64 // W, (3, 3), kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L7 Conv
model.add(Conv2D(64 // W, (3, 3), kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Flatten())
# L8 Dense
model.add(GaussianNoise(NOISE))
model.add(Dense(256, kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# L9 Dense
model.add(Dense(128, kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.50))
# L10
model.add(Dense(64, kernel_initializer= initializer, kernel_regularizer=keras.regularizers.l2(L2)))
model.add(BatchNormalization())
# L11 Output
model.add(Dense(OUTPUT_CLASSES))
model.add(Activation('softmax', dtype='float32'))

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

model.summary()

batch_logger_cb = DataUtil.BatchLogger(LOG_FP)
early_stopping_cb = EarlyStopping('val_loss', patience=10, restore_best_weights=True)

# best val_loss seen so far (previous sessions), so a resumed run never overwrites better weights
best_val = None
try:
    with open(LOG_FP) as fd:
        vals = [float(v) for v in re.findall(r"'val_loss': (?:np\.float\d*\()?([0-9.eE+-]+)", fd.read())]
    best_val = min(vals) if vals else None
except OSError:
    pass
backup_cb = keras.callbacks.BackupAndRestore(BACKUP_DIR, save_freq='epoch', delete_checkpoint=False)

model_save_cb = keras.callbacks.ModelCheckpoint(
    filepath= MODEL_CHECKPOINTS_FP,
    save_weights_only= True,
    monitor= 'val_loss',
    mode= 'min',
    save_best_only= True,
    initial_value_threshold= best_val
)

hist = model.fit(
    x = train_datagen, 
    batch_size = batch_size, 
    epochs= epochs, 
    validation_data= validation_datagen, 
    callbacks= [backup_cb, batch_logger_cb, model_save_cb, early_stopping_cb] + ([keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)] if os.environ.get('MV3_LRDECAY') == '1' else []),
    verbose= 2
)

# EarlyStopping restored the best (lowest val_loss) weights
model.save('./Models/Final-Model.hdf5')