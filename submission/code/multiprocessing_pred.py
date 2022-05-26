import ifxdaq
import tensorflow as tf
import processing
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import matplotlib.pyplot as plot
import numpy as np
from multiprocessing import Process, Queue
from threading import Thread
import concurrent.futures
import keyboard
import math
model=tf.keras.models.load_model('C:\\Users\\jimmy\\PycharmProjects\\hackathon\\examples\\weights\\newreal_model')
config_file = "radar_configs/RadarIfxBGT60.json"


number_of_frames = 10
import threading
import os


def task1():
    count = 0
    raw_data = []

    with RadarIfxAvian(config_file) as device:  # Initialize the radar with configurations
        # print(device.device_id)
        while(True):
            if(keyboard.is_pressed("q")):
                break
            raw_data = []
            sample=[]
            #count+=1
            for i_frame, frame in enumerate(device):  # Loop through the frames coming from the radar

                raw_data.append(np.squeeze(frame['radar'].data / (4095.0)))  # Dividing by 4095.0 to scale the data
                if (i_frame == number_of_frames - 1):
                    data = np.asarray(raw_data)
                    range_doppler_map = processing.processing_rangeDopplerData(data)

                    # del data
                    break

        # extended = np.concatenate((range_doppler_map[:][0], range_doppler_map[:][1], range_doppler_map[:][2]))
        # print(extended.shape)
            for s in range(4, number_of_frames - 1):
                sample.append(range_doppler_map[s][:][:][:] - range_doppler_map[s - 4][:][:][:])
            sample = np.array(sample)
            sample = np.real(sample)
            task2(sample)

            #print(model.predict(sample))
            # print(sample.shape)


def task2(sample):
    myarr=[]
    prediction = model.predict(sample)
    print(prediction.sum(axis=0).argmax())
    # w = np.array([0.25, 0.25, 0.25,0.25])
    # i = np.arange(4) + 1
    # m = (prediction.reshape((1, 5, 4)) == i.reshape((4, 1, 1)))
    # myarr = np.argmax(np.sum(m, axis=2).T * w, axis=1) + 1
    # print(myarr)

    print(model.predict(sample))



# if __name__ == "__main__":
#     # print ID of current process
#     print("ID of process running main program: {}".format(os.getpid()))
#
#     # print name of main thread
#     print("Main thread name: {}".format(threading.current_thread().name))
#
#     # creating threads
#     t1 = threading.Thread(target=task1, name='t1')
#     t2 = threading.Thread(target=task2, name='t2',args=(t1.) )
#
#     # starting threads
#     t1.start()
#     t2.start()
#
#     # wait until all threads finish
#     t1.join()
#     t2.join()


with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(task1)
    # return_value = future.result()
    # future2=executor.submit(task2,return_value)

# task1()
