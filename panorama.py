"""
panorma.py

Main program for running our static image stitcher and video
image stitcher based on various inputs.
"""
import argparse
import os
import sys
from random import shuffle

import cv2
import imutils

from static_stitcher import Stitcher


def parser_define():
    """
    Defines the parser for our program.

    Returns:
        an ArgumentParser object
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-imgd", "--image_dir", required=True, help="directory of images to stitch together")
    ap.add_argument("-r", "--rand", help="randomize the image list for testing arbitrary order", action="store_true")
    ap.add_argument("-t", "--test", help="if testing flag is set extra images will be displayed", action="store_true")
    ap.add_argument("-f", "--fall", help="if set will not use cv2 stitcher but our custom one", action="store_true")

    return ap


def get_image_list(directory):
    """
    Gets all images in the given directory.

    Params:
        directory (str): image directory

    Returns:
        a list of strings.
    """
    images = []
    for img in os.listdir(directory):
        path = os.path.join(directory, img)
        images.append(path)

    return images


def main():
    """
    Main program, takes in arguments and calls various methods for stitching and
    panorama creation.
    """
    ap = parser_define()
    args = ap.parse_args()

    images = get_image_list(args.image_dir)
    print("Found images: ", images)

    if args.rand:
        shuffle(images)
        print("Shuffled images: ", images)

    if len(images) < 2:
        sys.exit("There needs to be at least two images to stitch.")

    read_images = {}
    for image in images:
        img = cv2.imread(image)
        if args.test:
            cv2.imshow(image, imutils.resize(img, height=400))
        read_images[image] = img

    # create our static image stitcher and tell it not to use opencl
    static_stitcher = Stitcher(False)

    if not args.fall:
        print("Trying to use the cv2 stitcher...")
        result = static_stitcher.cv2_stitch(read_images)
    else:
        print("Forcing the use of our fall back stitcher...")
        result = static_stitcher.stitch_images(read_images)

        print("Success!")
        cv2.imshow('Result', imutils.resize(result, height=500))
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
