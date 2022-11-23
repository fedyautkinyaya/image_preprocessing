import cv2 as cv
import os
import numpy as np

'''Код преобразовывает входящие иображения из папки, путь к которой мы задаем. 
Данные преобразования проводятся для того, чтобы нейронная сеть точнее распознавал текст на изображениях'''



def adaptive_threshold(images_dir):
    images = os.listdir(images_dir)
    kernel = np.ones((1, 1), np.uint8)
    for i in range(len(images)):
        path = images_dir + images[i]
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 22)
        #img = cv.erode(img, kernel, iterations=1) # Erosion
        #img = cv.dilate(img, kernel, iterations=1) # Dilation
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel) # Opening (dilation + erosion)
        cv.imwrite(path, img)


images_path = input("input the path: ")
adaptive_threshold(images_path)

# '/home/fedyautkin/PycharmProjects/pythonProject4py379/pose_estimation/1/task1/'

