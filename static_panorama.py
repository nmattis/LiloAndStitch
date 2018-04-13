"""
static panorma.py
Nick Mattis

Following: https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
"""
import argparse
import os
import sys
from random import shuffle

import cv2
import imutils

from fallback_stitcher import FallBackStitcher


def get_image_list(directory):
    images = []
    for img in os.listdir(directory):
        path = os.path.join(directory, img)
        images.append(path)

    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="directory of images to stitch together")
    ap.add_argument("-r", "--rand", help="randomize the image list for testing arbitrary order", action="store_true")
    ap.add_argument("-t", "--test", help="if testing flag is set extra images will be displayed", action="store_true")
    ap.add_argument("-f", "--fall", help="if set will not use cv2 stitcher but our custom one", action="store_true")
    args = ap.parse_args()

    images = get_image_list(args.dir)
    print("Found images: ", images)

    if args.rand:
        shuffle(images)
        print("Shuffled images: ", images)

    if len(images) < 2:
        sys.exit("There needs to be at least two images to stitch.")

    read_images = []
    for image in images:
        img = cv2.imread(image)
        if args.test:
            cv2.imshow(image, imutils.resize(img, height=400))
        read_images.append(img)

    print("Creating stitchers...")
    stitcher = cv2.createStitcher(False)
    fallBack = FallBackStitcher(images)

    if not args.fall:
        print("Trying to use the cv2 stitcher...")
        result = stitcher.stitch(read_images)

        if result[1] is not None:
            print("Success!")
            cv2.imshow('Result', imutils.resize(result[1], height=500))
            cv2.waitKey(0)
        else:
            print("Not enough overlap for the cv2 stitcher, trying our fall back...")
            # we need to use our fall back stitcher
            result = fallBack.stitch()

            print("Success!")
            # cv2.imshow('Result', imutils.resize(result, height=500))
            # cv2.waitKey(0)
    else:
        print("Forcing the use of our fall back stitcher...")
        result = fallBack.stitch()

        print("Success!")
        # cv2.imshow('Result', imutils.resize(result, height=500))
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
