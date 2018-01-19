import os
import cv2
import numpy as np


dataset_path = 'D:\Bazy\JetsonHacksCherryCar\Fort3rdFloorDataSet\Fort3rdFloorDataSet~\TrainingIMG'
# original dataset at: https://drive.google.com/drive/folders/0Bx1XHaRXk3kSUGZuVnJfaWJiY2M?usp=sharing


def cherryCar_loadData_full(maxNumerOfSamples):
    x_full, y_full, fileNames = [], [], []
    for i in os.walk(dataset_path):
        (d, sub_dirs, fileName) = i
        fileNames.extend(fileName)

    sequence_fileName = []
    for fileName in fileNames:
        seqNum = int(fileName.split('_')[0])
        sequence_fileName.append((seqNum, fileName))
    sequence_fileName.sort() # order of images when reading from python is different than the one show in Windows Explorer

    for i, (seqNum, fileName) in enumerate(sequence_fileName):
        if i >= maxNumerOfSamples:
            break
        # img = cv2.imread(path+'/'+fname, 1)  # for black and white
        img = cv2.imread(dataset_path+'/'+fileName)
        img = cv2.resize(img, (200, 150), interpolation=cv2.INTER_CUBIC)

        # img_flip = cv2.flip(img, 0) # 0 - horizontal flip (1 - vertical flip)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for black and white
        # img2 = img[35:,:,:] # cropping top pixels
        # img = img[35:,:] for black and white
        # img = np.reshape(img, (115, 200, 1))  # for black and white

        _, timestamp, throttle, steeringAngle = fileName.split('_')
        timestamp, throttle, steeringAngle = int(timestamp), float(throttle), float(steeringAngle.split('.jpg')[0])
        print('seq: {}, timestamp: {}, throttle: {}, steering: {}'.format(seqNum, timestamp, throttle, steeringAngle))
        x_full.append(img)

        # y.append((steeringAngle, throttle))
        y_full.append(steeringAngle)  # only steering angle is important right now
        # x_full.append(img_flip) # for flipped images
        # y_full.append(1.0 - steeringAngle)  # for flipped images

    x_test = np.zeros(1)
    y_test = np.zeros(1)

    x_full_np = np.asarray(x_full)
    y_full_np = np.asarray(y_full)

    return x_full_np, y_full_np, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, _, _ = cherryCar_loadData_full(100)



