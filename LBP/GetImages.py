import cv2
from tqdm import tqdm
import glob
from skimage.transform import resize
import os

# the get_images() functions reads all images from the folder. On deafult it reads all images but you
# can feed it a different path to a subfolder to read fewer images for testing purposes.

def get_images(path="../AWEDataset/awe/**"):

    images={}

    for image_path in tqdm( sorted(glob.glob(os.path.join(path, "*.png"))) , desc="Reading images... "):
        label = (int(image_path[-10:-7])-1)*10 + int(image_path[-6:-4])
        image = cv2.imread(image_path, 0)
        resized_image = resize(image, (128, 128))
        images[label] = resized_image

    return images