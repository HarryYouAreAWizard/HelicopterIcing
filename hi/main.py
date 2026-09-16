



import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from pathlib import Path

from video import filter_frame, change_color_representation, extract_pixels_in_mask, normalize_brightness, set_frame
from mask_creation import reduced_frame_polygon, create_mask

import mask_creation
import video
from ice_noice import get_video_frame_averaged_HSV, get_video_frame_FULL_HSV, silhouette_score_from_mean_HSVs 
import ice_noice
from pynput import mouse

figure_dir          = Path(os.getcwd()) / "figures"
data_dir            = Path(os.getcwd()) / "video-data"
polygon_dir         = Path(os.getcwd()) / "polygons"
case_videos_dir     = Path(os.getcwd()) / "case_videos"
segnet_output_dir   = Path(os.getcwd()) / "segnet_output"
scan_output_dir     = Path(os.getcwd()) / "scan_output"
# reduced_video_dir   = Path(os.getcwd()) / "reduced_video_data"


video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

# to avoid spam, even though the video works fine
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"


def main()->None:

    from convmodel import livestream_test

    livestream_test(0)


main()