# LiloAndStitch
CV Project for creating a panorama from a set of arbitrary videos from different positions of the same subject

To run the project just clone and run ```pip install -r requirements.txt```

All data sets used can be found in ```/images``` or ```/videos```

Various testing output results can be found in ```/results```

## Usage

For best results use a data set of images that have at least a 20-30% overlap in image content and ensure that they are named in such a way that they are ordered from left to right. This is important if you choose to use our fallback stitcher implementation as it does not handle an arbitrary order. If just going to use the cv2 stitcher than order of images does not matter just overlap does.


This program can stitch images, videos, and create a panorama from video.

1) To create a panorama from a set of static images run:

    ```python panorama.py -imgd <path to directory of images>```

2) To create a static panorama fromm a video file run:

    ```python panorama.py -vidf <video file for panorama gen>```

3) To create a video panorama from a set of video files run:

    ```python panorama.py -vidd <path to directory of video files>```

There are also various flags for operatoin you can do:

1) ```-t, --test``` will allow you to view various images during the process
2) ```-f, --fall``` this flag will force the use of our stitch algorithm instead of the opencv stitcher for static image panorama generation
3) ```-dp, --draw_pts``` will show the keypoints detected and matching keypoints of the various images given to the program
4) ```-ns, --use_sampling``` if set will use naive sampling when creating a static panorama from a video
5) ```-ds, --use_deltas``` if set will use delta change for sampling unique frames when creating a static panorama form a video

    ** Set both ```-ns``` and ```-ds``` to use the combination of the two when creating a static image from a video file

## Trouble Shooting

If when creating a video panorama you get no output file or it errors saying something about codecs you will need to manually change the VideoWriter object in video_stitcher.py to match the usable codec for your OS. Googling this for opencv will tell you valid codecs for whatever OS you are using. Change the CC value on line 99 and the output file extension on line 100:

```
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('output.avi',fourcc, 30.0, (max_width, max_height))
```
