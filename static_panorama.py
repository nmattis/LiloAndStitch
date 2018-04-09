"""
static panorma.py
Nick Mattis

Following: https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
"""
from stitcher import Stitcher
import argparse
import imutils
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=True, help="path to the first image")
    ap.add_argument("-s", "--second", required=True, help="path to the second image")
    args = vars(ap.parse_args())

    imgA = cv2.imread(args["first"])
    imgB = cv2.imread(args["second"])
    imgA = imutils.resize(imgA, width=400)
    imgB = imutils.resize(imgB, width=400)

    stitch = Stitcher()
    (result, vis) = stitch.stitch([imgA, imgB], showMatches=True)

    cv2.imshow("A", imgA)
    cv2.imshow("B", imgB)
    cv2.imshow("Keypoint", vis)
    cv2.imshow("Res", result)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
