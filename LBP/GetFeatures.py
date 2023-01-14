import itertools
import json
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import resize
from tqdm import tqdm
from skimage.feature import local_binary_pattern

from .LBP import LBP, LBP_histogram, LBP_max_diff


# The get_features() functions takes in the following paramters and calls the appropriate LBP function
# Paramteres:
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
# "shifted" uses the rotational invariant alghorithm that shifts all binary values of features to their
#           biggest value
def get_features(images, type="basicLBP", radius=1, samples=8, step=1, shifted=False):

    feature_vectors = []
    for image_key in tqdm( images , desc="Extracting features... "):
        
        if type == "histogram":
            feature_vector = LBP_histogram(images[image_key], shifted, radius, samples)
        elif type == "basicLBP":
            if shifted:
                feature_vector = LBP_max_diff(images[image_key], radius, samples).ravel().tolist()
            else:
                feature_vector = LBP(images[image_key], radius, samples, step).ravel().tolist()
        elif type == "scikit":
            feature_vector = local_binary_pattern(images[image_key], samples, radius).ravel().tolist()

        feature_vectors.append(feature_vector)

    
    return feature_vectors
