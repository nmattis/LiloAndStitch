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
    ap.add_argument("-vidf", "--vid_file", help="path to video to turn into a panorama")
    ap.add_argument("-vidd", "--vid_dir", help="directory of videos to stitch together")
    ap.add_argument("-t", "--test", help="if testing flag is set extra images will be displayed", action="store_true")
    ap.add_argument("-f", "--fall", help="if set will not use cv2 stitcher but our custom one", action="store_true")
    ap.add_argument("-ns", "--use_sampling", help="if set will use naive sampling for the video panoram", action="store_true")
    ap.add_argument("-ds", "--use_deltas", help="if set will use delta sampling for the video panoram", action="store_true")

    return ap


def get_list(directory):
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
    images = get_list(directory)
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
    static_stitcher = ImageStitcher(True)

    if not fallback_flag:
        result = static_stitcher.cv2_stitch(read_images)
        if result[1] is not None:
            print("Success!")
            cv2.imshow('Result', imutils.resize(result[1], height=600))
            cv2.waitKey(0)
        else:
            print("Failed!")
    else:
        print("Forcing the use of our fall back stitcher...")
        result = static_stitcher.stitch_images(read_images)
        print("Success!")
        cv2.imshow('Result', imutils.resize(result, height=600))
        cv2.waitKey(0)


def video_stitching(video, isSweeping, use_sampling, use_deltas):
    """
    Performs panorama creation from a video file.

    Params:
        video (str): path to video file
    """
    if not use_sampling and not use_deltas and isSweeping:
        sys.exit("You must specify a sampling type to run the sweeping video stitch.")

    video_stitcher = VideoStitcher(7, 0.15, True)
    if isSweeping:
        result = video_stitcher.sweep_stitch(video, use_sampling, use_deltas)
        if result[1] is not None:
            print("Success!")
            cv2.imshow('Result', imutils.resize(result[1], height=600))
            cv2.waitKey(0)
        else:
            print("Failed!")
    else:
        videos = get_list(video)
        video_stitcher.static_stitch(videos)


def main():
    """
    Main program, takes in arguments and calls various methods for stitching and
    panorama creation.
    """
    ap = parser_define()
    args = ap.parse_args()

    if not args.image_dir and not args.vid_file and not args.vid_dir:
        sys.exit("Must specify images or video to use to create a panorama.")

    if args.image_dir:
        print("Performing static image stitching...")
        image_stitching(args.image_dir, args.test, args.fall)
        sys.exit("Done! Terminating Program.")

    if args.vid_file:
        print("Performing video stitching...")
        video_stitching(args.vid_file, True, args.use_sampling, args.use_deltas)
        sys.exit("Done! Terminating Program.")

    if args.vid_dir:
        print("Performing static video stitching...")
        video_stitching(args.vid_dir, False, args.use_sampling, args.use_deltas)
        sys.exit("Done! Terminating Program.")


if __name__ == '__main__':
    main()
