import cv2
import numpy as np

def drawWhiteHexagon(image, center_x, center_y, side_l):
    r3 = np.sqrt(3)
    hexagon_arr = np.array([[[center_x, center_y-side_l], [center_x-r3/2*side_l, center_y-side_l/2],
                             [center_x-r3/2*side_l, center_y+side_l/2], [center_x, center_y+side_l],
                             [center_x+r3/2*side_l, center_y+side_l/2], [center_x+r3/2*side_l,center_y-side_l/2]]],
                           dtype=np.int32)
    cv2.fillPoly(image, hexagon_arr, 1)
    return image

def generateHoneycombMask(image_size, hex_side_l, white_ratio, upper_left_ref, rotation, data_type):
    mask_size = int(np.ceil(np.sqrt(2) * max(image_size) + 3 * hex_side_l))
    white_hex_l = hex_side_l * white_ratio
    mask = np.zeros([mask_size, mask_size], dtype=data_type)
    r3 = np.sqrt(3)
    start_x = upper_left_ref[0] + r3/2*hex_side_l
    start_y = upper_left_ref[1] + hex_side_l
    center_x = start_x
    center_y = start_y
    while(center_y < mask_size - 2.5 * hex_side_l):
        while(center_x < mask_size - r3 * hex_side_l):
            mask = drawWhiteHexagon(mask, center_x, center_y, white_hex_l)
            center_x2 = center_x + r3/2*hex_side_l
            center_y2 = center_y + 1.5*hex_side_l
            mask = drawWhiteHexagon(mask, center_x2, center_y2, white_hex_l)
            center_x = center_x + r3 * hex_side_l
        center_y = center_y + 3 * hex_side_l
        center_x = start_x
    rotation_matrix = cv2.getRotationMatrix2D((mask_size/2, mask_size/2),rotation,1)
    mask_r = cv2.warpAffine(mask, rotation_matrix, (mask_size, mask_size))
    roi_start = int((2-np.sqrt(2))/4*mask_size)
    roi_end_x = roi_start + image_size[0]
    roi_end_y = roi_start + image_size[1]
    return mask_r[roi_start:roi_end_x, roi_start:roi_end_y]



# cx = 50
# cy = 50
# sl = 16
# sr3o2 = np.sqrt(3)/2
# hexagon1 = np.array([[[cx, cy-sl], [cx-sr3o2*sl, cy-sl/2], [cx-sr3o2*sl, cy+sl/2], \
#                       [cx, cy+sl],[cx+sr3o2*sl,cy+sl/2],[cx+sr3o2*sl,cy-sl/2]]], dtype=np.int32)
# im1 = np.zeros([240, 320], dtype=np.uint8)
# cv2.fillPoly(im1, hexagon1, 255)
# im1 = drawWhiteHexagon(im1,cx,cy,16)
# im1 = generateHoneycombMask([240, 360], 20, 0.8, [0,0], 80, "uint8")
# print im1.dtype
# cv2.imshow('image',im1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

