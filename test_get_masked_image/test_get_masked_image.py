


import src.mask_creation as mask_creation
import src.video as video
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

figure_dir = Path(os.getcwd()) / "figures"
data_dir = Path(os.getcwd()) / "data"

video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

# to avoid spam, even though the video works finde
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

start_frame_idx = 5000
def main()->None:
    # load the video
    cap = cv2.VideoCapture(data_dir / video_filename)

    # load the mask
    mask = mask_creation.get_mask(cap, plot_mask=True, figure_dir=figure_dir)


    # go to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    # ret the image
    ret, frame = cap.read()
    if not ret:
        print(f"{ret = }")
        return 

    # use the mask
    frame = frame * mask[:, :, None]

    # normalize brightness
    frame = video.normalize_brightness(frame, mask)

    # save image
    cv2.imwrite(figure_dir / "image.png", frame)

    print(f"Saved image at {figure_dir}/image.png")


main()