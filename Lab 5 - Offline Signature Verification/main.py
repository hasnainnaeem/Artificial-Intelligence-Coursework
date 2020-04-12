"""
AI - Lab 5 - Offline Signature Verification
Written by: M. Hasnain Naeem
"""

# imports
import math
import os
from PIL import Image
import path
import numpy as np
import pandas as pd
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
    try:
        cx = cx // n
        cy = cy // n
    except ZeroDivisionError:
        pass
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

    # coordinates of top left segment
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
    segment_imgs = [top_left_seg, top_right_seg, bottom_left_seg, bottom_right_seg]
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


def convert_to_bnw(dir, saving_dir):
    dir = os.path.join(os.getcwd(), dir)
    saving_dir = os.path.join(os.getcwd(), saving_dir)

    for img_name in os.listdir(dir):
        img_path = os.path.join(dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # convert to black and white
        img[img > 220] = 255
        img[img != 255] = 0

        new_img_path = os.path.join(saving_dir, img_name.split(".")[0] + ".jpg")

        # save
        cv2.imwrite(new_img_path, img)


def get_segments(img):
    bounded_img, bounded_box_dims = bounded_box(img)
    centroid_dims = centroid(img)
    segment_imgs, coordinates = segments(img, centroid_dims, bounded_box_dims)
    return segment_imgs, coordinates


def black_pixels(img):
    black_pixel_count = 0
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            if val == 0:
                black_pixel_count += 1
    return black_pixel_count


def normalized_size(img, black_pixel_count):
    try:
        norm_size = (img.shape[0] * img.shape[1]) / black_pixel_count
    except ZeroDivisionError:
        norm_size = 0
    return norm_size


def centroid_inclination(img, centroid):
    try:
        angle = math.atan(centroid[1] / centroid[0])
    except ZeroDivisionError:
        angle = 0
    return angle


def normalized_pixel_inclination(img):
    angle_sum = 0
    black_pixel_count = 0
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            if val == 0:
                black_pixel_count += 1
                try:
                    angle_sum += math.atan(i / j)
                except ZeroDivisionError:
                    angle_sum = 0
    try:
        normalized_angle = angle_sum / black_pixel_count
    except ZeroDivisionError:
        normalized_angle = 0
    return normalized_angle, black_pixel_count


def lab5():
    ref_dir = r"Dataset_4NSigComp2010\TestSet\Reference"
    ref_dir = os.path.join(os.getcwd(), ref_dir)

    file_dir = "Files"
    bnw_dir = os.path.join(os.getcwd(), file_dir, "BnW")
    if not os.path.exists(bnw_dir):
        os.makedirs(bnw_dir)
    # convert images to B & W
    print("Converting images to Black & White.")
    convert_to_bnw(ref_dir, bnw_dir)

    img_names = os.listdir(bnw_dir)

    details = ["centroid", "transitions", "aspect_ratio", "inclination_angle",
               "norm_black_pixels_angle", "no_of_black_pixels", "normalized_size"]
    indices = pd.MultiIndex.from_product([img_names, details])
    df = pd.DataFrame(index=indices, columns={i: [] for i in range(64)})

    for img_name in img_names:
        print("Processing image: " + img_name)
        # open file and convert to grey scale
        filename = os.path.join(bnw_dir, img_name)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        first_level_segments, _ = get_segments(img)
        second_level_segments = []
        for segment in first_level_segments:
            second_level_segments.extend(get_segments(segment)[0])
        third_level_segments = []
        for segment in second_level_segments:
            third_level_segments.extend(get_segments(segment)[0])

        all_segments = []
        # store 64 segments. Ignore 20 segments from the 3rd level
        all_segments.extend(first_level_segments)  # len = 4
        all_segments.extend(second_level_segments)  # len = 16
        all_segments.extend(third_level_segments[:44])  # 44
        # Path to store all the segments
        segments_path = os.path.join(os.getcwd(), "Files\\Processed")
        segments_path = os.path.join(segments_path, img_name.split(".")[0])

        if not os.path.exists(segments_path):
            os.makedirs(segments_path)
        for i, segment in enumerate(all_segments):
            df[i].loc[img_name]["transitions"] = black_to_white_trans(segment)
            df[i].loc[img_name]["aspect_ratio"] = segment.shape[1] // segment.shape[0]
            df[i].loc[img_name]["centroid"] = centroid(segment)
            df[i].loc[img_name]["inclination_angle"] = centroid_inclination(segment, df[i].loc[img_name]["centroid"])

            norm_b_angle, no_of_black_pixels = normalized_pixel_inclination(segment)

            df[i].loc[img_name]["norm_black_pixels_angle"] = norm_b_angle
            df[i].loc[img_name]["no_of_black_pixels"] = no_of_black_pixels
            df[i].loc[img_name]["normalized_size"] = normalized_size(segment, no_of_black_pixels)

            segment_path = os.path.join(segments_path, img_name.split(".")[0] + "_" + str(i) + ".jpg")
            cv2.imwrite(segment_path, segment)

    print("Centroid, transition and aspect ratio details are saved in directory:")
    print(os.path.join(os.getcwd(), "dataset_details.xlsx"))
    df.to_excel("dataset_details.xlsx")


# Main
lab5()
