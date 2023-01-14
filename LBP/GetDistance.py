import numpy as np
import json
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics import pairwise_distances

# the function get_distance_mtx() calculates the distances between all feature vectors and returns
# them in a distance matrix. It can compute the euclidean or cosine distance depending on the incoming
# parameter. Before switching to the pairwise_distances() function from sklearn library I had my own 
# function for computing distances but it took way to much time.

def get_distance_mtx(data, method="euclidean"):

    distances = pairwise_distances(data, metric=method)
    # setting the diagonal elements to "a lot" to avoid images being closest to themselves. Or better said
    # to avoid that impacting the results.
    distances[distances<0.0001] = np.inf

    return distances

