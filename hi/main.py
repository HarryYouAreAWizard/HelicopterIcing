



import numpy as np
import matplotlib.pyplot as plt
# import cv2
import os
# import torch

from pathlib import Path

import convmodel
import volumeestimation
# import mask_creation
# import ice_noice
figure_dir          = Path(os.getcwd()) / "figures"
data_dir            = Path(os.getcwd()) / "video-data"
polygon_dir         = Path(os.getcwd()) / "polygons"
case_videos_dir     = Path(os.getcwd()) / "case_videos"
segnet_output_dir   = Path(os.getcwd()) / "segnet_output"
scan_output_dir     = Path(os.getcwd()) / "scan_output"
# reduced_video_dir   = Path(os.getcwd()) / "reduced_video_data"
label_data_dir      = Path(os.getcwd()) / "label_data"
model_weights_dir   = Path(os.getcwd()) / "model_weights"
volume_dir          = Path(os.getcwd()) / "volume"


video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

# to avoid spam, even though the video works fine
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"


def main()->None:

    fig, ax=plt.subplots()
    apparent_thicknesses = np.load(volume_dir / "apparent pixel thickness.npy")
    ax.plot(apparent_thicknesses)

    from scipy.ndimage import uniform_filter

    filtered = uniform_filter(apparent_thicknesses, size=len(apparent_thicknesses)//100)
    ax.plot(filtered)

    ax.legend()
    
    fig.savefig(figure_dir / f"Observed pixel thickness {video_filename}.png")


    print(f"loading model...")
    model_entries = convmodel.load_model(load_weights=True)
    model, loss_func, optimizer, scheduler = model_entries

    print(f"loading data...")
    data = convmodel.load_data_with_labels()

    print(f"starting training...")
    convmodel.train(
        model_entries=model_entries,
        data=data,
        batch_size=25,
        epochs=100,
        ignore_scheduler=True
    )

    print(f"Testing live...")
    convmodel.livestream_test(model, 8000)
    return
    print(f"Running inference on video")
    apparent_thicknesses = volumeestimation.run_apparent_thickness(videocapture=video_filename, model=model)
    np.save(volume_dir / "apparent pixel thickness.npy", apparent_thicknesses)

    


main()