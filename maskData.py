import cv2
from random import *
import numpy as np
from generateHoneycomb import generateHoneycombMask
import os
"""
    Main function for synthetizing data. The script will read in all pictures in a certain directory, add user defined 
    honeycomb pattern to them and save them with an extra 'masked' suffix.
    
    Required input: 
        image_path: directory path to images
        hexagon_size: side length of hexagons in the mask
        white_ratio: ratio between white (unblocked part) and black (blocked part) in each hexagon
        suffix: suffix of the image file type. default is '.JPEG'
    Other input (will be randomly assigned)
        pattern_shift: how much the mash will shift in both direction (since it's a periodic pattern, shift < side length)
        rotation_angle: the angular orientation of mask
"""
image_path = "/media/ron/New Volume/Downloads/ILSVRC/Data/DET/test/"

hexagon_size = 24
white_ratio = 0.9
suffix = '.JPEG'
for file_name in os.listdir(image_path):
    if file_name.endswith(suffix):
        pattern_shift = [int(random()*hexagon_size/2),int(random()*hexagon_size/2)]
        rotation_angle = random()*90
        img = cv2.imread(image_path + file_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size_x, size_y = gray_img.shape
        img_mask = generateHoneycombMask([size_x, size_y], hexagon_size, white_ratio, pattern_shift, rotation_angle, gray_img.dtype)
        gray_img_masked = cv2.multiply(img_mask, gray_img)
        cv2.imwrite(image_path + file_name[:-len(suffix)]+'masked' + suffix, gray_img_masked)

