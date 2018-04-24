"""
video_stitcher.py

Class for performing panorama creation from video.
"""

import cv2
import imutils
import numpy as np

class VideoStitcher:
    def __init__(self, sample_rate, use_opencl):
        self.sample_rate = sample_rate
        cv2.ocl.setUseOpenCL(use_opencl)
        self.cv2_stitcher = cv2.createStitcher(False)

    def stitch(self, video):
        vid_cap = cv2.VideoCapture(video)
        count = 0
        frames = []
        while True:
            success, image = vid_cap.read()
            if success and count % self.sample_rate == 0:
                rows, cols, _ = image.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
                des = cv2.warpAffine(image, M, (cols, rows))
                frames.append(des)
                print("Read a new frame: ", success)
            count += 1
            if success == False:
                break
        
        print("Grabbed frames")
        print("Using cv2 stitcher to create panorama...")
        result = self.cv2_stitcher.stitch(frames)
        if result[1] is not None:
            print("Success!")
            cv2.imshow('Result', imutils.resize(result[1], height=500))
            cv2.waitKey(0)
        else:
            print("Failure")
