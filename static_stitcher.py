"""
stitcher.py

Reference: https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
"""
import cv2
import imutils
import numpy as np

class Stitcher:
    def __init__(self, use_opencl):
        # determine if we are using OpenCV v3
        self.isv3 = imutils.is_cv3()
        # create the CV2 stitcher for use if we can
        cv2.ocl.setUseOpenCL(use_opencl)
        self.cv2_stitcher = cv2.createStitcher(False)

    def cv2_stitch(self, images):
        """
        Attempts using the default cv2 stitcher as its better and faster, if no
        resulting stitch created (usually do to overlap not being within 20-30%)
        then it falls back to using our hand written stitcher which is  slower
        and not as good.

        Params:
            images (dict): dict of image name and numpy array read in images
        """
        print("Trying to use the cv2 stitcher...")
        result = self.cv2_stitcher.stitch(list(images.values()))

        if result[1] is not None:
            print("Success!")
            cv2.imshow('Result', imutils.resize(result[1], height=500))
            cv2.waitKey(0)
        else:
            print("Not enough overlap for the cv2 stitcher, trying our fall back...")
            # we need to use our fall back stitcher
            result = self.stitch_images(images)

            print("Success!")
            cv2.imshow('Result', imutils.resize(result, height=500))
            cv2.waitKey(0)

    def stitch_images(self, images):
        """
        Runs our handwritten stithcer with multiple images, stitches the first
        two images and then takes the result and stitches that to the next one
        and so on.

        Params:
            images (dict): dict of image name and numpy array read in images

        Returns the numpy array of the stitched image result.
        """
        imageA, imageB = list(images.keys())[:2]
        imgA = images[imageA]
        imgB = images[imageB]

        result = self.stitch([imgA, imgB])
        for img in list(images.keys())[2:]:
            result = self.stitch([result, images[img]])

        return result

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        """
        Detects the features and keypoints using SIFT, matches those keypoints
        between the images to calculate and Homography matrix, then applies
        a warpAffine transform (scale, rotation, etc) to the images creating
        a result that is the two images stitched together.

        Params:
            images (lst): list of numpy array image data for two images
            ratio (float): David Lowe's ratio test for false-pos feat matching
            reprojThresh (float): max pixel threshold room for RANSAC
            showMatches (bool): True will draw the matching keypoints

        Returns:
            a stitched result of the two images
        """
        (imgB, imgA) = images
        # take each image, detect keypoints and extract
        # local invariant descriptors from them
        print("Detecting Keypoints and Features...")
        (kpsA, featuresA) = self.detectAndDescribe(imgA)
        (kpsB, featuresB) = self.detectAndDescribe(imgB)

        # match features between the two images
        print("Matching Keypoints and Features...")
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective wrap to stitch the images together
        (matches, H, status) = M
        # print("imgA shape", imgA.shape)
        # print("imgB shape", imgB.shape)
        print("Applying the transformation...")
        result = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], max(imgA.shape[0], imgB.shape[0])))
        result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB

        # check to see if the keypoint matchess should be visualized
        if showMatches:
            vis = self.drawMatches(imgB, imgA, kpsB, kpsA, matches, status)

            # return a tuple of the stiched image and the visualization
            return (result, vis)
        
        # return the stiched image
        return result

    def detectAndDescribe(self, image):
        """
        Grayscales the image and uses SIFT to find keypoints and
        features of the image.

        Params:
            image (numpy array): image data
        
        Returns:
            the keypoints and features of the image
        """
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
        """
        Goes through the keypoints and applies knn matching and then uses
        David Lowe's reatio test and ransac to get matches within our tolerances.
        After that calculates the homography matrix we need for transformation.

        Params:
            kpsA (numpy array): image A keypoints
            kpsB (numpy array): image B keypoints
            featuresA (numpy array): image A features
            featuresB (numpy array): image B features
            ratio (float): David Lowe's ratio test for false-pos feat matching
            reprojThresh (float): max pixel threshold room for RANSAC
            
        Returns:
            None if we don't have enough matches to compute the homography,
            or the keypoints/features that matched, the calcuated homography
            matrix, and status given by findHomography.
        """
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
        """
        For drawing the matches over the two images to see how we did.

        Params:
            imgA (numpy array): image A data
            imgB (numpy array): image B data
            kpsA (numpy array): keypoints for image A
            kpsB (numpy array): keypoints for image B
            matches (numpy array): matches we found
            status (numpy array): output mask from findHomography

        Returns:
            resulting visualization of the image keypoints
        """
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
