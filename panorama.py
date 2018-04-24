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

from static_stitcher import ImageStitcher
from video_stitcher import VideoStitcher


def parser_define():
    """
    Defines the parser for our program.

    Returns:
        an ArgumentParser object
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-imgd", "--image_dir", help="directory of images to stitch together")
    ap.add_argument("-vidd", "--vid_file", help="path to video to turn into a panorama")
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


def image_stitching(directory, test_flag, fallback_flag):
    """
    Performs static image stitching given a directory of image
    files to stitch together.

    Params:
        directory (str): path to images
    """
    images = get_image_list(directory)
    print("Found images: ", images)

    if len(images) < 2:
        sys.exit("There needs to be at least two images to stitch.")

    read_images = {}
    for image in images:
        img = cv2.imread(image)
        if test_flag:
            cv2.imshow(image, imutils.resize(img, height=400))
        read_images[image] = img

    # create our static image stitcher and tell it not to use opencl
    static_stitcher = ImageStitcher(False)

    if not fallback_flag:
        print("Trying to use the cv2 stitcher...")
        static_stitcher.cv2_stitch(read_images)
    else:
        print("Forcing the use of our fall back stitcher...")
        static_stitcher.stitch_images(read_images)


def video_stitching(video):
    """
    Performs panorama creation from a video file.

    Params:
        video (str): path to video file
    """
    video_stitcher = VideoStitcher(50, False)
    video_stitcher.stitch(video)


def main():
    """
    Main program, takes in arguments and calls various methods for stitching and
    panorama creation.
    """
    ap = parser_define()
    args = ap.parse_args()

    if not args.image_dir and not args.vid_file:
        sys.exit("Must specify images or video to use to create a panorama.")

    if args.image_dir:
        print("Performing static image stitching...")
        image_stitching(args.image_dir, args.test, args.fall)
        sys.exit("Done! Terminating Program.")

    if args.vid_file:
        print("Performing video stitching...")
        video_stitching(args.vid_file)
        sys.exit("Done! Terminating Program.")


if __name__ == '__main__':
    main()
