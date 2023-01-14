from matplotlib import pyplot as plt
import numpy as np
from LBP import *
from GetFeatures import get_features
from GetDistance import *
from GetScore import *
from GetImages import get_images
import cv2
from skimage.transform import resize
from skimage.feature import local_binary_pattern

# in main() function use singleRun() to test single combinations of different implemented functions.
# "type" will determine which LBP type the code will run from the following options:
#   - "basicLBP": the LBP that I created
#   - "histogram": my LBP use
#   - "scikit": local_binary_pattern function from the skimage library
# "method" determines which distance the alhorithm will compute between feature vectors. You can choose between:
#   - "euclidean" and
#   - "cosine"
# "radius" determines the radius of the circle in which we will be comparing pixels
# "samples" determines the number of pixel we'll be comparing
# "step" determines how many pixels we'll skip in the image (1 means we take every pixel)
def singleRun(type="basicLBP", method="euclidean", radius=1, samples=8, step=1):
    images = get_images()
    feature_vectors = get_features(images, type=type, radius=radius, samples=samples, step=step)
    distanceMtx  = get_distance_mtx(feature_vectors, method=method)
    accuracy = get_scores(distanceMtx)
    saveTo = "Results/" + type + "_" + method + "_r-" + str(radius) + "_smp-" + str(samples) + "_step-" + str(step) + ".txt"
    with open(saveTo, 'w') as f:
        f.write(f"Type of LBP: {type}\n")
        f.write(f"Calculating distance with: {method}\n")
        f.write(f"radius={radius}, samples={samples} --> Accuracy: {accuracy}\n")

    f.close()


# in main() function use runAll() to get results of all combination of implemented functions. 
# This combinates differents LBP methods (no pixel-by-pixel) at different radius, 
# number of compared pixels and method for computing distance (euclidean or cosine)
#
# After running the runAll() function you can ran the GetScore.py file in the terminal to get
# the plots used in my report. If you change the name of the .json file with the results a change is also
# needed in the GetPlot.py file.
def runAll():
    images = get_images()
    types = ["histogram", "basicLBP", "scikit"]
    methods = ["euclidean", "cosine"]
    # samples=8
    resultJson = {}
    step = 2
    with open("Results/final.txt", 'w') as f:
        for type in types:
            resultJson[type] = {}
            f.write(f"\nType of LBP: {type}\n")
            for method in methods:
                resultJson[type][method] = {}
                f.write(f"  Calculating distance with: {method}\n")
                for radius in range(1, 13, 2):
                    resultJson[type][method][str(radius)] = {}
                    if radius == 1:
                        samples = 8
                        print("Case:", type, method, str(radius), str(samples))
                        feature_vectors = get_features(images, type=type, radius=radius, samples=samples, step=step)
                        distanceMtx = get_distance_mtx(feature_vectors, method=method)
                        accuracy = get_scores(distanceMtx)

                        resultJson[type][method][str(radius)][str(samples)] = accuracy
                        f.write(f"      radius={radius}, samples={samples} --> Accuracy: {accuracy}\n")
                    else:
                        for samples in range(8, 15, 2):
                            print("Case:", type, method, str(radius), str(samples))
                            feature_vectors = get_features(images, type=type, radius=radius, samples=samples, step=step)
                            distanceMtx = get_distance_mtx(feature_vectors, method=method)
                            accuracy = get_scores(distanceMtx)

                            resultJson[type][method][str(radius)][str(samples)] = accuracy
                            f.write(f"      radius={radius}, samples={samples} --> Accuracy: {accuracy}\n")

    f.close()

    with open("Results/final.json", "w") as outfile:
        json.dump(resultJson, outfile)

# in main() function use pixelByPixel() to get pixel by pixel comparison of all images computed with both 
# euclidean and cosine distances.
def pixelByPixel():
    images = get_images()
    feature_vectors = []
    for image_key in tqdm( images , desc="Extracting features... "):
        feature_vector = images[image_key].ravel().tolist()
        feature_vectors.append(feature_vector)

    methods = ["euclidean", "cosine"]
    with open("Results/pixel-by-pixel.txt", 'w') as f:
        f.write(f"Pixel-by-pixel comparison\n")
        for method in methods:
            distanceMtx = get_distance_mtx(feature_vectors, method=method)
            accuracy = get_scores(distanceMtx)
            f.write(f"  Calculating distance with: {method}\n")
            f.write(f"      Accuracy: {accuracy}\n")



def main():
    # Uncomment lines for the above functions that you wish to run.
    # pixelByPixel()
    singleRun(type="histogram", radius=4, samples=12, step=3)
    # runAll()

main()
