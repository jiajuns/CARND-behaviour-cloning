import csv
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.layers import normalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from keras import optimizers


images = []
measurements = []
datapath = []
datapath.append(os.getcwd() + '/data/normal/driving_log.csv')
datapath.append(os.getcwd() + '/data/normal2/driving_log.csv')
datapath.append(os.getcwd() + '/data/change_boarder/driving_log.csv')
datapath.append(os.getcwd() + '/data/additional/driving_log.csv')
datapath.append(os.getcwd() + '/data/normal3/driving_log.csv')
datapath.append(os.getcwd() + '/data/normal4/driving_log.csv')
datapath.append(os.getcwd() + '/data/newtrack1/driving_log.csv')

for file_path in datapath:
    with open(file_path) as csvfile:
        lines = csv.reader(csvfile)
        for line in lines:
            file_path = line[0]
            prob = np.random.random()
            # file_path = line[0].split('/')
            if abs(float(line[3])) <= 0.001 and prob < 0.5:
                continue
            # file_path = os.path.join(os.getcwd(), 'sample_data', file_path[0], file_path[1])
            # print(file_path)
            image = cv2.imread(file_path)
            images.append(image)
            images.append(np.fliplr(image))
            steering_center = float(line[3])
            measurements.append(steering_center)
            measurements.append(-steering_center)
        csvfile.close()

X_train = np.array(images)
y_train = np.array(measurements)

dropout_rate = 0.5
activation_type = 'relu'
lamb = 1e-5

model = Sequential()
model.add(Cropping2D(cropping=((50,10), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(filters=24, kernel_size=(7, 7), strides=(2, 2), padding='valid', activation=activation_type))
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation=activation_type))
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation=activation_type))
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=activation_type))
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=activation_type))
model.add(Flatten())

model.add(Dense(1743, activation=activation_type, kernel_regularizer=regularizers.l2(lamb)))
model.add(Dense(200, activation=activation_type, kernel_regularizer=regularizers.l2(lamb)))
model.add(Dense(100, activation=activation_type, kernel_regularizer=regularizers.l2(lamb)))
model.add(Dense(20, activation=activation_type, kernel_regularizer=regularizers.l2(lamb)))
model.add(Dense(1))

rmsprop = optimizers.RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop)
model.summary()

# earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20, callbacks=[checkpointer, reduce_lr])
