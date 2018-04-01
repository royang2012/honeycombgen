import unittest
import cv2
from generateHoneycomb import generateHoneycombMask
from random import *
class TestHoneycombGen(unittest.TestCase):
    def test_mask_gen(self):
        self.assertEqual(generateHoneycombMask([240, 360], 20, 0.8, [0,0], 80, "uint8").dtype, "uint8")

    def test_mask_img(self):
        image_path = "/media/ron/New Volume/Downloads/ILSVRC/Data/DET/test/"
        hexagon_size = 24
        white_ratio = 0.9
        image_name = "ILSVRC2017_test_00000001.JPEG"
        img = cv2.imread(image_path + image_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size_x, size_y = gray_img.shape
        img_mask = generateHoneycombMask([size_x, size_y], hexagon_size, white_ratio, [int(random()*hexagon_size/2),int(random()*hexagon_size/2)], random()*90, gray_img.dtype)
        gray_img_masked = cv2.multiply(img_mask, gray_img)
        cv2.imshow('image',gray_img_masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ori_img_size = img.shape
        masked_img_size = gray_img_masked.shape
        self.assertEqual(ori_img_size[0:1], masked_img_size[0:1])
    def test_avg_blur(self):
        image_name = "messi.jpg"
        white_ratio = 0.8
        img = cv2.imread(image_name)
        hexagon_size = 10
        pattern_shift = [int(random()*hexagon_size/2),int(random()*hexagon_size/2)]
        rotation_angle = random()*90
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.blur(gray_img, (20,20))
        size_x, size_y = gray_img.shape
        img_mask = generateHoneycombMask([size_x, size_y], hexagon_size, white_ratio, pattern_shift, rotation_angle, gray_img.dtype)
        gray_img_masked = cv2.multiply(img_mask, blur_img)
        cv2.imshow('image',gray_img_masked)
        cv2.waitKey(0)

if __name__ == '__main__':
    unittest.main()