import ifxdaq
import processing
import numpy as np
from PIL import Image
import statistics
import tensorflow as tf
from tensorflow.keras.layers import LSTM
import os
import zipfile
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Conv3D,MaxPooling3D,Conv1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler

checkpoint_path='C:\\Users\\jimmy\\PycharmProjects\\hackathon\\examples\\weights\\weights-best.hdf5'


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

number_of_frames = 2000
#
# print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import matplotlib.pyplot as plot

mergef1 = "C:\\Users\\jimmy\Downloads\\challenge\\raw_processed_data\\examples\\processed\\oneyp"
mergef3 = "C:\\Users\\jimmy\Downloads\\challenge\\raw_processed_data\\examples\\processed\\threeyp"
mergef2 = "C:\\Users\\jimmy\Downloads\\challenge\\raw_processed_data\examples\\processed\\twoyp"
mergef0 = "C:\\Users\\jimmy\Downloads\\challenge\\raw_processed_data\\examples\\processed\\zeroyp"

x1 = []
x2 = []
x3 = []
x0 = []
count =0
for x in os.listdir(mergef1):
        path = mergef1 + '/' + x
        x1.append(np.load(path))
        # for i in x1:
        #   print(i)
for x in os.listdir(mergef2):
        path = mergef2 + '/' + x
        x2.append(np.load(path))
        # for i in x1:
        #   print(i)
for x in os.listdir(mergef3):
        path = mergef3 + '/' + x
        x3.append(np.load(path))
        # for i in x1:
        #   print(i)
for x in os.listdir(mergef0):
        path = mergef0 + '/' + x
        x0.append(np.load(path))
        # for i in x1:
        #   print(i)

x1=np.array(x1)
x2=np.array(x2)
x3=np.array(x3)
x0=np.array(x0)
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x0.shape)

# x1m = []
# x2m = []
# x3m = []
# x4m = []
sample1 = []
sample2 = []
sample3 = []
sample0 = []

for s in range(6,number_of_frames-1):
            sample1.append(x1[s][:][:][:] - x1[s-6][:][:][:])
            sample2.append(x2[s][:][:][:] - x2[s-6][:][:][:])
            sample3.append(x3[s][:][:][:] - x3[s-6][:][:][:])
            sample0.append(x0[s][:][:][:] - x0[s-6][:][:][:])


sample1=np.array(sample1)
sample2=np.array(sample2)
sample3=np.array(sample3)
sample0=np.array(sample0)
print(sample1[0].shape)
y1 = np.full(1993,1)
y2 = np.full(1993,2)
y3 = np.full(1993,3)
y0 = np.full(1993,0)

y0 =y0.reshape(-1,1)
y1 =y1.reshape(-1,1)
y2 =y2.reshape(-1,1)
y3 =y3.reshape(-1,1)
print(y0.shape)
# X = np.concatenate((x0,x1,x2,x3))
X = np.concatenate((sample0,sample1,sample2,sample3))

mean_antenna = []

for s in range(number_of_frames):
             mean_antenna.append((X[s][0][:][:]+X[s][1][:][:]+X[s][2][:][:])/3)
mean_antenna = np.array(mean_antenna)

X = np.real(X)
print(X.shape)
print("mean")
# Xmerge = []
# cm = 0
# for i in X:
#     if (cm%5==0):
#         Xmerge.append(np.concatenate(X[cm],X[cm-1],X[cm-2],X[cm-3],X[cm-4]))
#     cm += 1
# Xmerge = np.array(Xmerge)
# print(Xmerge.shape)

Y = np.concatenate((y0,y1,y2,y3))

print(X.shape)

from tensorflow.keras.utils import to_categorical
# print(Y)

# encoder = OneHotEncoder(sparse=False)
# Y = encoder.fit_transform(Y)
Y=to_categorical(Y)
print(Y)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,
                                                                                            Y,
                                                                                            test_size=0.2,
                                                                                            shuffle=True,
                                                                                            random_state=39)


print(X_train.shape)

model = Sequential()

activation='relu'
model.add(MaxPooling2D(input_shape=[3,64,64]))
model.add((Conv1D(64, 3, activation = activation)))

model.add((BatchNormalization)())
model.add((Conv1D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform')))
model.add(BatchNormalization())
model.add((Conv1D(16, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform')))
model.add(BatchNormalization())
# model.add((MaxPooling2D()))
#
#
# model.add((Conv2D(16, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform')))
# # model.add(BatchNormalization())
# model.add((MaxPooling2D()))


# model.add(Flatten())
# model.add(Dense(128, activation = activation, kernel_initializer = 'he_uniform'))
# model.add(Dense(4, activation = 'softmax'))

# after having Conv2D...
# model.add(
#     TimeDistributed(
#         Conv2D(64, (3,3), activation='relu'),
#         input_shape=(5, 3, 64, 64) # 5 images...
#     )
# )
# model.add(
#     TimeDistributed(
#         Conv2D(64, (3,3), activation='relu')
#     )
# )
# We need to have only one dimension per output
# to insert them to the LSTM layer - Flatten or use Pooling

# previous layer gives 5 outputs, Keras will make the job
# to configure LSTM inputs shape (5, ...)
# model.add(
#     LSTM(1024, activation='relu', return_sequences=False)
# )

model.add(Flatten())
model.add(Dense(64, activation = activation, kernel_initializer = 'he_uniform'))
model.add(Dense(4, activation = 'softmax'))

model.add(Dense(4, activation='softmax'))

Adam = tf.keras.optimizers.Adam(learning_rate=0.0003) #learning rate

model.compile(optimizer = Adam ,loss = 'categorical_crossentropy', metrics = ['accuracy'] )
print(model.summary())
# model.load_weights(checkpoint_path)
# #=================================train model==================================

#Csv logger
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)
#ModelCheckpoint callback saves best model
checkpoint = ModelCheckpoint("C:\\Users\\jimmy\\PycharmProjects\\hackathon\\examples\\weights-best-60acc.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#Early stopping
early_stop = EarlyStopping(monitor="val_loss", patience =3 , verbose=1)
#callbacks list
callbacks_list=[early_stop , log_csv, checkpoint]


history = model.fit(X_train, y_train, epochs=8, validation_data=(X_test, y_test),callbacks=callbacks_list)
model.save('C:\\Users\\jimmy\\PycharmProjects\\hackathon\\examples\\weights\\temp_model')

model.save('C:\\Users\\jimmy\\PycharmProjects\\hackathon\\examples\\weights\\sixtyacc_model')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'm', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training & validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'm', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training & validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#
# config_file = "radar_configs/RadarIfxBGT60.json"
# count=0
# while(count<12):
#     count+=1
#     number_of_frames = 1
#
#     ## Run this to understand the current radar settings better
#     import json
#     with open(config_file) as json_file:
#         c = json.load(json_file)["device_config"]["fmcw_single_shape"]
#         chirp_duration = c["num_samples_per_chirp"]/c['sample_rate_Hz']
#         frame_duration = (chirp_duration + c['chirp_repetition_time_s']) * c['num_chirps_per_frame']
#         print("With the current configuration, the radar will send out " + str(c['num_chirps_per_frame']) + \
#               ' signals with varying frequency ("chirps") between ' + str(c['start_frequency_Hz']/1e9) + " GHz and " + \
#               str(c['end_frequency_Hz']/1e9) + " GHz.")
#         print('Each chirp will consist of ' + str(c["num_samples_per_chirp"]) + ' ADC measurements of the IF signal ("samples").')
#         print('A chirp takes ' + str(chirp_duration*1e6) + ' microseconds and the delay between the chirps is ' + str(c['chirp_repetition_time_s']*1e6) +' microseconds.')
#         print('With a total frame duration of ' + str(frame_duration*1e3) + ' milliseconds and a delay of ' + str(c['frame_repetition_time_s']*1e3) + ' milliseconds between the frame we get a frame rate of ' + str(1/(frame_duration + c['frame_repetition_time_s'])) + ' radar frames per second.')
#
#         raw_data = []
#         sample = []
#         with RadarIfxAvian(config_file) as device:  # Initialize the radar with configurations
#             print(device.device_id)
#             for i_frame, frame in enumerate(device):  # Loop through the frames coming from the radar
#
#                 raw_data.append(np.squeeze(frame['radar'].data / (4095.0)))  # Dividing by 4095.0 to scale the data
#                 if (i_frame == number_of_frames - 1):
#                     data = np.asarray(raw_data)
#                     range_doppler_map = processing.processing_rangeDopplerData(data)
#                     # del data
#                     break
#
#
#
#     print("predict")
#
#     prediction = model.predict(range_doppler_map)
#
#     print(prediction)
#
#
#     if prediction[0][0] > 0.5:
#         print( "0 malakas")
#     elif prediction[0][1] > 0.5:
#         print("1 malakas")
#     elif prediction[0][2] > 0.5:
#         print("2 malakas")
#     else:
#         print("3 malakas")