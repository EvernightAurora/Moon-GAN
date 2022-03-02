import cv2
import os
import numpy as np


TRUE_LABEL_DIR = 'Moons'


def load_datasets():
    ret = []
    for i in os.listdir(TRUE_LABEL_DIR):
        fn = os.path.join(TRUE_LABEL_DIR, i)
        img = cv2.imread(fn)
        ret.append(img.astype(np.float32)/128 - 1)
    return np.array(ret)