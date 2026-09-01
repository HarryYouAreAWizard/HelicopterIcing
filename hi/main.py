



import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from pathlib import Path

import mask_creation
import video
import case_seperation

figure_dir = Path(os.getcwd()) / "figures"
data_dir = Path(os.getcwd()) / "video-data"
case_videos_dir = Path(os.getcwd()) / "case_videos"

video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

# to avoid spam, even though the video works finde
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

def create_new_video_in_interval(video_in, video_out, interval):
    start, end = interval
    return 

start_frame_idx = 5000
def main()->None:
    # load the video
    cap = cv2.VideoCapture(data_dir / video_filename)

    # load the mask
    mask = mask_creation.get_mask(cap, plot_mask=True, figure_dir=figure_dir)
    if isinstance(mask, int):
        print("mask creation failed")
        return 

    # go to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    # read the image
    ret, frame = cap.read()
    if not ret:
        return 

    frame = video.filter_frame(frame, mask)
    frame = video.normalize_brightness(frame, mask)

    # save image
    cv2.imwrite(figure_dir / "image.png", frame)

    print(f"Saved image at {figure_dir}/image.png")

    
    print(f"\nFinding indicies...")
    frame_indicies_ice, frame_indicies_noice = case_seperation.get_ice_indices(cap, mask, end_idx=None)
    frame_indicies_ice = np.array(frame_indicies_ice)
    frame_indicies_noice = np.array(frame_indicies_noice)

    print(f"\nSaving indicies...")
    np.save(case_videos_dir / "frame_indicies_ice.npy", frame_indicies_ice)
    np.save(case_videos_dir / "frame_indicies_noice.npy", frame_indicies_noice)

    print(f"{np.shape(frame_indicies_ice) = }")
    print(f"{np.shape(frame_indicies_noice) = }")

main()