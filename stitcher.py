"""
stitcher.py
Nick Mattis

Following: https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
"""
import cv2
import imutils
import numpy as np

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # take each image, detect keypoints and extract
        # local invariant descriptors from them
        (imgB, imgA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imgA)
        (kpsB, featuresB) = self.detectAndDescribe(imgB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective wrap to stitch the images together
        (matches, H, status) = M
        result = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
        result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB

        # check to see if the keypoint matchess should be visualized
        if showMatches:
            vis = self.drawMatches(imgA, imgB, kpsA, kpsB, matches, status)

            # return a tuple of the stiched image and the visualization
            return (result, vis)
        
        # return the stiched image
        return result

    def detectAndDescribe(self, image):
        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see the cv version
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise since not v3
        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NuPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            # ensure the distance is within a certain ratio of each other
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            return (matches, H, status)

        return None

    def drawMatches(self, imgA, imgB, kpsA, kpsB, matches, status):
        (hA, wA) = imgA.shape[:2]
        (hB, wB) = imgB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imgA
        vis[0:hB, wA:] = imgB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[queryIdx][0]) + wA, int(kpsB[queryIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        
        return vis
