import glob
import itertools
import os
import cv2
import numpy as np
import PIL
from matplotlib import pyplot as plt

# the LBP funcion accepts a greyscale image and calculates its features accoring to the parameters:
# "radius" determines the radius of the circle in which we will be comparing pixels
# "samples" determines the number of pixel we'll be comparing
# "step" determines how many pixels we'll skip in the image (1 means we take every pixel)

def LBP (gray_image, radius=1, samples=8, step=1):

    coor=[]
    for i in range(samples):
        x = -radius * np.sin(2*i*np.pi/samples)
        y = radius * np.cos(2*i*np.pi/samples)
        coor.append([round(x), round(y)])

    paddedImage = cv2.copyMakeBorder(gray_image,radius,radius,radius,radius,cv2.BORDER_REPLICATE)
    features = np.zeros_like(paddedImage[radius:-radius:step, radius:-radius:step])

    for row_indeks, row in enumerate(paddedImage[radius:-radius:step, radius:-radius:step]):
        for column_indeks, pixel in enumerate(row):
            feature = 0
            # print("checking pixel at: (", row_indeks, ",", column_indeks, ")")
            for power, position in enumerate(coor):
                # print("comparing to pixel at: (", row_indeks+position[0], ",", column_indeks+position[1], ")")
                x = row_indeks*step+radius
                y = column_indeks*step+radius
                if pixel < paddedImage[x+position[0], y+position[1]]:
                    feature += np.power(2, power)
            features[row_indeks, column_indeks] = feature

    return features


# the LBP_max_diff() function is an extension of the upper LBP() function. It adds rotational invariance
# by shitfing the binary value of every feature to its biggest value.

def LBP_max_diff(gray_image, radius=1, samples=8):
    
    coor=[]
    for i in range(samples):
        x = -radius * np.sin(2*i*np.pi/samples)
        y = radius * np.cos(2*i*np.pi/samples)
        coor.append([round(x), round(y)])

    features = np.zeros_like(gray_image)
    paddedImage = cv2.copyMakeBorder(gray_image,radius,radius,radius,radius,cv2.BORDER_REPLICATE)
    for row_indeks, row in enumerate(paddedImage[radius:-radius, radius:-radius], start=radius):
        for column_indeks, pixel in enumerate(row, start=radius):
            binary = ''
            for position in coor:
                if pixel < paddedImage[row_indeks+position[0], column_indeks+position[1]]:
                    binary += '1'
                else:
                    binary += '0'

            maxDiff = int('0b'+binary, 2)
            for i in range(1, len(binary)):
                shifted = '0b' + binary[i:] + binary[:i]
                if int(shifted, 2) > maxDiff:
                    maxDiff = int(shifted, 2)

            features[row_indeks-radius, column_indeks-radius] = maxDiff

    return features


# the LBP_histogram() is another extension of the first LBP() function. 
# 1. It takes the features computed by the LBP() function, 
# 2. splits them into 64 areas (8 intervals by height and 8 by width: in our constant 128x128 
# images that comes to 64 16x16 areas), 
# 3. computes histograms for each area,
# and 4. concatenates them into a single feature vector.

def LBP_histogram(gray_image, shifted, radius=1, samples=8, step=1):

    if shifted:
        features = LBP_max_diff(gray_image, radius, samples)

    else:
        features = LBP(gray_image, radius, samples, step)


    feature_vector = []

    height, width = features.shape
    stepHeight = round(height/8)
    stepWidth = round(width/8)
    for i in range(0, height-stepHeight+1, stepHeight):
        for j in range(0, width-stepWidth+1, stepWidth):
            partialHist, bins = np.histogram(features[i:i+stepHeight, j:j+stepWidth].ravel(), np.arange(np.power(2, samples)))
            feature_vector = np.concatenate((feature_vector, partialHist))

    return feature_vector

