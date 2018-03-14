import cv2
from random import *
import numpy as np
from generateHoneycomb import generateHoneycombMask
import os
image_path = "/media/ron/New Volume/Downloads/ILSVRC/Data/DET/test/"
hexagon_size = 24
white_ratio = 0.9
for file_name in os.listdir(image_path):
    if file_name.endswith(".JPEG"):
        img = cv2.imread(image_path + file_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size_x, size_y = gray_img.shape
        img_mask = generateHoneycombMask([size_x, size_y], hexagon_size, white_ratio, [int(random()*hexagon_size/2),int(random()*hexagon_size/2)], random()*90, gray_img.dtype)
        gray_img_masked = cv2.multiply(img_mask, gray_img)
        cv2.imwrite(image_path + file_name[:-5]+'masked.JPEG', gray_img_masked)

