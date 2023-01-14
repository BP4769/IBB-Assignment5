import numpy as np
from tqdm import tqdm

# the get_score() function calculates the accuracy of the algorithm by going through 
# the matrix of distances bewteen all features vectors of images and finding which are closest
# If images belong to the same class it is considered as a successful clasification.

def get_scores(distances):

    success = 0
    fail = 0
    length = len(distances)

    for i in tqdm( range(length), desc="Calulating score..."):
        minDistance = np.min(distances[i])
        minIndex = np.where(distances[i] == minDistance)[0][0]
        if minIndex // 10 == i // 10:
            success += 1
            # print("Image ", i, " is closest to image ", minIndex)
        else:
            fail += 1

    return (success/length)*100
