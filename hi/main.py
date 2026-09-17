



import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch

from pathlib import Path

import convmodel
import mask_creation
import ice_noice
figure_dir          = Path(os.getcwd()) / "figures"
data_dir            = Path(os.getcwd()) / "video-data"
polygon_dir         = Path(os.getcwd()) / "polygons"
case_videos_dir     = Path(os.getcwd()) / "case_videos"
segnet_output_dir   = Path(os.getcwd()) / "segnet_output"
scan_output_dir     = Path(os.getcwd()) / "scan_output"
# reduced_video_dir   = Path(os.getcwd()) / "reduced_video_data"
label_data_dir      = Path(os.getcwd()) / "label_data"
model_weights_dir   = Path(os.getcwd()) / "model_weights"


video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

# to avoid spam, even though the video works fine
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"


def main()->None:

    print(f"loading model...")
    model_entries = convmodel.load_model(load_weights=False)
    model, loss_func, optimizer, scheduler = model_entries
    model.mean = np.float32(0.0)
    model.std = np.float32(1.0)

    convmodel.livestream_test(model, 0)
    return

    print(f"loading data...")
    data = convmodel.load_data_with_labels()


    print(f"starting training...")
    convmodel.train(
        model_entries=model_entries,
        data=data,
        batch_size=20,
        epochs=25
    )

main()