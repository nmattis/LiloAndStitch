"""
video_panorama.py
Sid Nutulapati

"""

import argparse
import cv2
import imutils

SAMPLE_RATE = 10


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
      success, image = vidcap.read()
      frames.append(image)
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

  if result[1] is not None:
    print("Success!")
    cv2.imshow('Result', imutils.resize(result[1], height=500))
    cv2.waitKey(0)
  else:
    print("Failure :(")
    return


if __name__ == "__main__":
  main()
