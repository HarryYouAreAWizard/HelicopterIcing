



import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas
from pathlib import Path

from video import filter_frame, change_color_representation, extract_pixels_in_mask, normalize_brightness, set_frame
from mask_creation import reduced_frame_polygon, create_mask

import mask_creation
import video

from pynput import mouse

figure_dir          = Path(os.getcwd()) / "figures"
# data_dir            = Path(os.getcwd()) / "video-data"
data_dir            = Path("..") / "data"
polygon_dir         = Path(os.getcwd()) / "polygons"
case_videos_dir     = Path(os.getcwd()) / "case_videos"
segnet_output_dir   = Path(os.getcwd()) / "segnet_output"
scan_output_dir     = Path(os.getcwd()) / "scan_output"
reduced_video_dir   = Path(os.getcwd()) / "reduced_video_data"


video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

# 1. Define the mouse callback function
def click_event(event, x, y, flags, param):
    # Check if the event is a left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: X: {x}, Y: {y}")
        storage.append((x, y))     
    


def main()->None:
    cap = cv2.VideoCapture(data_dir / video_filename)
    pipe_mask = mask_creation.get_mask(cap, "pipe_small", plot_mask=True, figure_dir=figure_dir)
    print(f"{figure_dir.__str__() = }")
    global storage
    storage = []
    saved_training_data = {}
    cv2.namedWindow("RF")
    cv2.setMouseCallback("RF", click_event)

    try:
        df = pandas.read_csv("training_data.csv")
    except pandas.errors.EmptyDataError:
        df = pandas.DataFrame()
    print(f"{df.columns = }")

    start = 10000
    end = 40000
    i = start
    set_frame(cap, i)
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret: break

        if str(i) in df.columns:
            i += 1
            continue

        frame = filter_frame(frame, pipe_mask)
        gray = change_color_representation(frame, cv2.COLOR_BGR2GRAY)

        nonzero = cv2.findNonZero(pipe_mask).reshape((-1, 2))

        x_min = nonzero[:, 0].min()
        x_max = nonzero[:, 0].max()
        y_min = nonzero[:, 1].min()
        y_max = nonzero[:, 1].max()

        relevant_frame = gray[y_min:y_max, x_min:x_max]

        print(f"Showing frame {i}")
        cv2.imshow("RF", relevant_frame)
        key = cv2.waitKey(0)
        saved_training_data[i] = storage
        print(f"{saved_training_data = }")
        temp = pandas.Series(storage, name=str(i))
        df = pandas.concat([df, temp], axis=1)
        df.to_csv("training_data.csv")
        print(f"{df = }")

        storage = []
        i += 1

        if i%10==0:
            i = np.random.randint(start, end)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # quit
        if key == ord("q"):
            exit()
            break

        elif key == ord("s"):
            i = int(input("new frame: "))
            set_frame(cap, i)

        if i == end:
            break

    cv2.destroyAllWindows()



    return 
main()