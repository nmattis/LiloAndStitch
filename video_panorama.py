"""
video_panorama.py
Sid Nutulapati

"""

import argparse
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 15


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-v", "--vid", required=True, help="path to video to turn into a panorama")
  args = ap.parse_args()
  vid = args.vid

  vidcap = cv2.VideoCapture(vid)
  count = 0
  frames = []
  while True:
    success, image = vidcap.read()
    if success and count % SAMPLE_RATE == 0:
      rows, cols, _ = image.shape
      M = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
      dest = cv2.warpAffine(image, M, (cols, rows))
      frames.append(dest)
      print('Read a new frame: ', success)
    count += 1
    if success == False:
      break

  print("Grabbed frames")
  print("Creating stitchers...")
  cv2.ocl.setUseOpenCL(False)
  stitcher = cv2.createStitcher(False)

  print("Trying to use the cv2 stitcher...")
  result = stitcher.stitch(frames)
  # plot('Frame something', imutils.resize(frames[2], height=500))
  if result[1] is not None:
    print("Success!")
    cv2.imshow("Result w/ Sample Rate {}".format(SAMPLE_RATE), imutils.resize(result[1], height=500))
    cv2.waitKey(0)
  else:
    print("Failure :(")
    return
  # plt.show()
  return


def plot(title, image):
  plt.figure(title)
  plt.axis("off")
  plt.imshow(image)
  plt.draw()



if __name__ == "__main__":
  main()
