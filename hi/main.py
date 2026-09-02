



import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch

from pathlib import Path

import mask_creation
import video
import ice_noice

from segnet import SegNet

figure_dir = Path(os.getcwd()) / "figures"
data_dir = Path(os.getcwd()) / "video-data"
case_videos_dir = Path(os.getcwd()) / "case_videos"
segnet_output_dir = Path(os.getcwd()) / "segnet_output"

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

    
    # print(f"\nFinding indicies...")
    # frame_indicies_ice, frame_indicies_noice = case_seperation.get_ice_indices(cap, mask, end_idx=None)
    # frame_indicies_ice = np.array(frame_indicies_ice)
    # frame_indicies_noice = np.array(frame_indicies_noice)

    # print(f"\nSaving indicies...")
    # np.save(case_videos_dir / "frame_indicies_ice.npy", frame_indicies_ice)
    # np.save(case_videos_dir / "frame_indicies_noice.npy", frame_indicies_noice)
    print("Loading indicies, test...")
    frame_indicies_ice = np.load(case_videos_dir / "frame_indicies_ice.npy")
    frame_indicies_noice = np.load(case_videos_dir / "frame_indicies_noice.npy")
    print(f"{np.shape(frame_indicies_ice) = }")
    print(f"{np.shape(frame_indicies_noice) = }")

    
    print("Initializing model...")
    model = SegNet(in_chn=3, out_chn=2)

    print("Reformating image...")
    video.set_frame(cap, 10000)
    ret, frame = cap.read()
    if not ret: print("Error")

    # remove beckground
    frame = video.filter_frame(frame, mask)                     # use mask
    frame = video.normalize_brightness(frame, mask)             # normalize with LAB
    frame = frame[frame.shape[0]//2:, frame.shape[1]//2:, :]    # cut to lower right
    cv2.imwrite(figure_dir / "image.png", frame)

    # prepare image for SegNet
    image = torch.tensor(frame)
    image = image.to(torch.float32) / 255.0
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)

    # run and save output
    print("Running inference on image")
    out = model.forward(image)
    torch.save(out, segnet_output_dir / "out")

    print("Saving segnet output as image")
    out = torch.load(segnet_output_dir / "out")
    out = out.squeeze(0)
    out = out.permute(1, 2, 0)
    out = (out * 255).to(torch.uint8)
    out = out.cpu().detach().numpy()
    print(f"{out.shape = }")
    frame = out[:, :, 0]
    print(f"{frame.shape = }")
    print(frame)
    cv2.imwrite(figure_dir / "SegNet output.png", frame)



main()