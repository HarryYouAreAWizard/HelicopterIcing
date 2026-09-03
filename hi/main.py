



import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from pathlib import Path

import mask_creation
import video
from ice_noice import get_video_frame_averaged_HSV, silhouette_score_from_mean_HSVs


figure_dir          = Path(os.getcwd()) / "figures"
data_dir            = Path(os.getcwd()) / "video-data"
polygon_dir         = Path(os.getcwd()) / "polygons"
case_videos_dir     = Path(os.getcwd()) / "case_videos"
segnet_output_dir   = Path(os.getcwd()) / "segnet_output"
scan_output_dir     = Path(os.getcwd()) / "scan_output"
reduced_video_dir   = Path(os.getcwd()) / "reduced_video_data"


video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

# to avoid spam, even though the video works fine
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"



def run_HSV_extraction(cap, mask):
    print("Getting frame average HSV...")
    HSV = get_video_frame_averaged_HSV(cap, mask)
    np.save(scan_output_dir / "HSV_springfield.npy", HSV)

def run_silhouette_score(skip):
    HSV = np.load(scan_output_dir / "HSV_springfield.npy")
    
    scan_center    = 0.095
    scan_halfwidth = 0.05
    thresholds = np.linspace(scan_center - scan_halfwidth, scan_center + scan_halfwidth, 100)

    print(f"Calculating silhouette scores for all threshold and all frames")
    silhouette_scores = silhouette_score_from_mean_HSVs(HSV, thresholds, skip=skip)

    np.save(scan_output_dir / "thresholds_springfield", thresholds)
    np.save(scan_output_dir / "silhouette_scores_springfield", silhouette_scores)
    np.save(scan_output_dir / "HSV_springfield", HSV)
    # np.save(scan_output_dir / "ice_mask_springfield", ice_mask)

    return thresholds, silhouette_scores


def main()->None:
    # # load the video
    # cap = cv2.VideoCapture(data_dir / video_filename)
    # # load the mask
    # mask = mask_creation.get_mask(cap, plot_mask=True, figure_dir=figure_dir)
    # if isinstance(mask, int):
    #     print("mask creation failed")
    #     return 


    print("Finding silhouette scores...")
    thresholds, silhouette_scores = run_silhouette_score(skip=5)

    fig, ax=plt.subplots()
    ax.plot(thresholds, silhouette_scores)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Optimizing icing threshold")
    fig.tight_layout()
    fig.savefig(figure_dir / "thresholds-silhouette.png")



main()