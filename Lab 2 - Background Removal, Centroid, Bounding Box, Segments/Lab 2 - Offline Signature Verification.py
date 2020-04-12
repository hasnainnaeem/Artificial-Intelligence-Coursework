"""
AI - Lab 2 - Offline Signature Verification
Written by: M. Hasnain Naeem
"""

# imports
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# functions
def show_image(img):
    # show image using OpenCV; extra lines are for jupyter notebook
    cv2.imshow("Gray Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bounded_box(bin_img):
    width = bin_img.shape[1] - 1
    height = bin_img.shape[0] - 1

    left = width
    right = 0
    top = height
    bottom = 0

    for x in range(0, width):
        for y in range(0, height):
            color = bin_img[y, x]
            if color == 0:
                if x > right:
                    right = x
                if x < left:
                    left = x
                if y > bottom:
                    bottom = y
                if y < top:
                    top = y

    bounded_box_dims = (left, right, top, bottom)
    bounded_img = bin_img[top: bottom, left:right]
    return (bounded_img, bounded_box_dims)


def draw_bounded_box(img, bounded_box_dims):
    img_with_box = img
    for i in range(bounded_box_dims[0], bounded_box_dims[1] + 1):
        img_with_box[bounded_box_dims[2]][i] = 0
        img_with_box[bounded_box_dims[3]][i] = 0

    for i in range(bounded_box_dims[2], bounded_box_dims[3] + 1):
        img_with_box[i][bounded_box_dims[0]] = 0
        img_with_box[i][bounded_box_dims[1]] = 0
    return img_with_box


def centroid(bin_img):
    width = bin_img.shape[1] - 1
    height = bin_img.shape[0] - 1
    cx = 0
    cy = 0
    n = 0
    for x in range(0, width):
        for y in range(0, height):
            color = bin_img[y, x]
            if color == 0:
                cx = cx + x
                cy = cy + y
                n = n + 1
    cx = cx // n
    cy = cy // n
    centroid_dims = (cx, cy)
    return centroid_dims


def segments(bin_img, centroid_dims, bounded_box_dims):
    """
    Get 4 segments from a image.

    Parameters:

    big_img: binary image with 1 color channel
    centroid: centroid tuple
    bounded_box: bounding box tuple with arrangement: (left, right, top, bottom)

    returns:

    segment_imgs: list of segments
    coordinates: list of coordinates of each segment
    """

    (left, right, top, bottom) = bounded_box_dims
    (cx, cy) = centroid_dims
    # coordinates of image segment
    top_left_coords = ((top, cy), (left, cx))
    # image segment
    top_left_seg = bin_img[top:cy + 1, left:cx + 1]
    # similarly for other segments:
    top_right_coords = ((top, cy), (cx, right))
    top_right_seg = bin_img[top:cy + 1, cx:right + 1]

    bottom_left_coords = ((cy, bottom), (left, cx))
    bottom_left_seg = bin_img[cy:bottom + 1, left:cx + 1]

    bottom_right_coords = ((cy, bottom), (cx, right))
    bottom_right_seg = bin_img[cy:bottom + 1, cx:right + 1]

    # list of segments
    segment_imgs = (top_left_seg, top_right_seg, bottom_left_seg, bottom_right_seg)
    # list of coordinates
    coordinates = (top_left_coords, top_right_coords, bottom_left_coords, bottom_right_coords)

    return segment_imgs, coordinates


def draw_segment_lines(img, segment_coordinates):
    (top_left_coords, top_right_coords, bottom_left_coords, bottom_right_coords) = segment_coordinates

    img_with_lines = img
    for i in range(top_left_coords[0][0], top_left_coords[0][1]):
        img_with_lines[i][top_left_coords[1][1]] = 0

    for i in range(bottom_left_coords[0][0], bottom_left_coords[0][1]):
        img_with_lines[i][bottom_left_coords[1][1]] = 0

    for i in range(top_left_coords[1][0], top_left_coords[1][1]):
        img_with_lines[top_left_coords[0][1]][i] = 0

    for i in range(bottom_right_coords[1][0], bottom_right_coords[1][1]):
        img_with_lines[bottom_right_coords[0][0]][i] = 0

    return img_with_lines


def black_to_white_trans(img):
    height = img.shape[0]
    width = img.shape[1]

    prev = img[0, 0]
    n = 0
    for x in range(1, width):
        for y in range(1, height):
            curr = img[y, x]
            if curr == 1 and prev == 0:
                n += 1
            prev = curr
    return n


def __main__():
    # creating image object
    sign_img = Image.open(r"Files/sample_sign.jpg")

    # using 8-bit pixel convertion to b&w
    sign_img_l = sign_img.convert("L")

    sign_img_l.save(r"Files/sample_sign_bw.jpg")

    file_dir = "Files/"

    # open file and convert to grey scale
    filename = os.path.join(file_dir, "sample_sign_bw.jpg")
    img = cv2.imread(filename)[:, :, 0]  # keep only one color dimension

    print("Shape of image:")
    print(img.shape)
    # show_image(img)

    # normalize
    img = img / 255
    show_image(img)

    # to analyze the pixel values
    hist, bin_edges = np.histogram(img)
    plt.bar(bin_edges[:-1], hist, width=1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.show()

    # convert to black and white
    img[img > .5] = 1
    img[img != 1] = 0

    # Task 1
    bounded_img, bounded_box_dims = bounded_box(img)
    show_image(bounded_img)
    # find and show image with bounding box
    img_with_box = draw_bounded_box(img, bounded_box_dims)
    show_image(img_with_box)

    # Task 2
    centroid_dims = centroid(img)
    cx, cy = centroid_dims
    print("Centroid is: ({0}, {1})".format(cx, cy))

    # Task 3
    # get list of segment images & their coordinates in original image
    segment_imgs, coordinates = segments(img, centroid_dims, bounded_box_dims)

    img_with_lines = draw_segment_lines(img, coordinates)
    show_image(img_with_lines)

    show_image(segment_imgs[0])  # top left
    show_image(segment_imgs[1])  # top right
    show_image(segment_imgs[2])  # bottom left
    show_image(segment_imgs[3])  # bottom right

    # task 4
    tl_trans = black_to_white_trans(segment_imgs[0])
    print("Tansitions in TL: {0}".format(tl_trans))
    tr_trans = black_to_white_trans(segment_imgs[1])
    print("Tansitions in TR: {0}".format(tr_trans))
    bl_trans = black_to_white_trans(segment_imgs[2])
    print("Tansitions in BL: {0}".format(bl_trans))
    br_trans = black_to_white_trans(segment_imgs[3])
    print("Tansitions in BR: {0}".format(br_trans))

    T = (tl_trans, tr_trans, bl_trans, br_trans)
    print()

    # Printing results of all tasks
    print("Value of B: " + str(bounded_box_dims))
    print("Value of C: " + str(centroid_dims))
    print("Value of T: " + str(T))


# Main
__main__()
