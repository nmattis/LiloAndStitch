import cv2
import imutils
import numpy as np

class FallBackStitcher:
    def __init__(self, images, prune_ratio=0.75):
        self.images = images
        self.prune_ratio = prune_ratio
        # surf faster than sift so use it
        # 400 = Hessian Threshold
        self.surf = cv2.xfeatures2d.SURF_create(400)

    def stitch(self):
        self.figOutOrder(500)
        # img_info = {}
        # for image in self.images:
        #     img_info[image] = self.detectKeyPoints(image)

        # for img1 in img_info.keys():
        #     for img2 in img_info.keys():
        #         if img1 == img2:
        #             continue

        #         info1 = img_info[img1]
        #         info2 = img_info[img2]
        #         match = self.matchPoints(info1["kps"], info2["kps"], info1["feat"], info2["feat"])

        #         print(img1, img2)
        #         print(len(match))
        #         break
        #     break

    def figOutOrder(self, cropMax=100):
        img_info_left = {}
        img_info_right = {}
        for image in self.images:
            img = cv2.imread(image)
            # crop to edges
            w, _, _ = img.shape
            img_info_left[image] = self.detectKeyPoints(img[:cropMax])
            img_info_right[image] = self.detectKeyPoints(img[(w - cropMax):])

        for img1, img1_val in img_info_left.items():
            for img2, img2_val in img_info_right.items():
                if img1 == img2 or img1_val["feat"] is None or img2_val["feat"] is None:
                    continue

                match = self.matchPoints(img1_val["feat"], img2_val["feat"])

                print("Lefts -> Rights ", img1, img2)
                print(len(match))

        for img1, img1_val in img_info_right.items():
            for img2, img2_val in img_info_left.items():
                if img1 == img2 or img1_val["feat"] is None or img2_val["feat"] is None:
                    continue

                match = self.matchPoints(img1_val["feat"], img2_val["feat"])

                print("Rights -> Lefts ", img1, img2)
                print(len(match))

    def detectKeyPoints(self, image):
        # img = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.surf.detectAndCompute(gray, None)

        # cv2.drawKeypoints(gray, keypoints, img)
        # cv2.imshow(image, img)
        # cv2.waitKey(0)

        return { "kps": keypoints, "feat": descriptors }

    def matchPoints(self, features1, features2):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(features1, features2, 2)
        
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.prune_ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        return matches
