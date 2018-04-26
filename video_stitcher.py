"""
video_stitcher.py

Class for performing panorama creation from video.
"""

import sys

import cv2
import imutils
import numpy as np


class VideoStitcher:
    def __init__(self, sample_rate, delta_thresh, use_opencl):
        self.sample_rate = sample_rate
        self.delta_thresh = delta_thresh
        cv2.ocl.setUseOpenCL(use_opencl)
        self.cv2_stitcher = cv2.createStitcher(False)

    def static_stitch(self, video_dir):
        pass

    def sweep_stitch(self, video, use_sampling, use_deltas):
        vid_cap = cv2.VideoCapture(video)

        if use_sampling and use_deltas:
            print("Delta Thresh = " + str(self.delta_thresh))
            print("Sampling Rate = " + str(self.sample_rate))
            d_sampled, d_total_frames, d_frames = self.frame_deltas(vid_cap)

            total_count = 0
            frames = []
            for frame in d_frames:
                if total_count % self.sample_rate == 0:
                    frames.append(frame)
                total_count += 1

            print("Sampled deltas: " + str(d_sampled))
            print("Final Sampling: " + str(len(frames)))
            print("Total frames: " + str(d_total_frames))

            if frames is None:
                sys.exit("Something when way wrong and we don't have any frames.")

            print("Using cv2 stitcher to create panorama...")
            result = self.cv2_stitcher.stitch(frames)

            return result

        if use_sampling:
            print("Sampling Frames Randomly... FS = " + str(self.sample_rate))
            sampled, total_frames, frames = self.frame_sampling(vid_cap)
            print("Total Sampled Frame Count: ", str(sampled))
            print("Total Count: ", str(total_frames))
        
        if use_deltas:
            print("Sampling Frames Using Deltas... Thresh = " + str(self.delta_thresh))
            sampled, total_frames, frames = self.frame_deltas(vid_cap)
            print("Total Sampled Frame Count: ", str(sampled))
            print("Total Count: ", str(total_frames))
        
        print("Done getting frames...")
        if frames is None:
            sys.exit("Something when way wrong and we don't have any frames.")

        print("Using cv2 stitcher to create panorama...")
        result = self.cv2_stitcher.stitch(frames)

        return result

    def frame_sampling(self, vid_cap):
        sample_count = 0
        total_count = 0
        frames = []
        while True:
            success, image = vid_cap.read()
            if success and total_count % self.sample_rate == 0:
                # this only here for the test video, should remove with a different vid
                rows, cols, _ = image.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
                frame = cv2.warpAffine(image, M, (cols, rows))
                frames.append(frame)
                sample_count += 1
            total_count += 1
            if success == False:
                break
        
        return sample_count, total_count, frames

    def frame_deltas(self, vid_cap):
        total_count = 0
        unique_count = 0
        prev_frame = None
        prev_frame_delta = 0
        frames = []
        while True:
            success, image = vid_cap.read()
            if success:
                total_count += 1
                # this only here for the test video, should remove with a different vid
                rows, cols, _ = image.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
                frame = cv2.warpAffine(image, M, (cols, rows))
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is None:
                    prev_frame = frame_gray
                    unique_count += 1
                else:
                    abs_diff = cv2.absdiff(prev_frame, frame_gray)
                    current_frame_delta = np.sum(abs_diff) / prev_frame.size
                    if prev_frame_delta == 0:
                        prev_frame_delta = current_frame_delta 
                    else:
                        delta_change = abs(prev_frame_delta - current_frame_delta)
                        if delta_change > self.delta_thresh:
                            print(delta_change)
                            frames.append(frame)
                            unique_count += 1
                        prev_frame_delta = current_frame_delta
                    prev_frame = frame_gray

            if success == False:
                break

        return unique_count, total_count, frames
