

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from pathlib import Path

from mask_creation import reduced_frame_polygon, create_mask

def manual_ice_finder(videocapture, mask, start):

    # open video at start
    i = start
    # cap.set(cv2.CAP_PROP_POS_FRAMES, i) # search up the frame
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, i) # search up the frame
    ret, frame = videocapture.read()

    n = 1
    while videocapture.isOpened():
        ret, frame = videocapture.read()
        if not ret:
            break

        cv2.imshow("..", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        if key == ord("b"):
            print(f'"start{n}": {i},')

        if key == ord("m"):
            print(f'"end{n}": {i},')
            n += 1
        elif key == ord(" "):
            k = cv2.waitKey(0)
            if k == ord(" "):
                continue
            elif k == ord("q"):
                break

        increase = 10
        for _ in range(increase - 1):
            # grab but dont decode
            videocapture.grab()

        i += increase

reduced_video_dir   = Path(os.getcwd()) / "reduced_video_data"
figure_dir          = Path(os.getcwd()) / "figures"
scan_output_dir     = Path(os.getcwd()) / "scan_output"

video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

red = cv2.VideoCapture(reduced_video_dir / video_filename)
_, frame = red.read()
polygon = reduced_frame_polygon(frame)
red_mask = create_mask(polygon, frame.shape)

# manual_ice_finder(red, red_mask, 0)



results = {
    "start1": 14900,
    "end1": 30100,
    "start2": 40750,
    "end2": 48450,
    "start3": 61250,
    "end3": 76750,
    "start4": 84850,
}
results = {
    "start1": 14500,
    "end1": 29800,
    "start2": 40750,
    "end2": 46550,
    "start3": 61300,
    "end3": 76600,
    "start4": 84800,
}

results = {
    "start1": 13500,
    "end1": 30280,
    "start2": 38120,
    "end2": 48940,
    "start3": 59520,
    "end3": 77100,
    "start4": 82390,
}

# this one matches with the original video
results = {
    "start1": 5500,
    "end1": 20500,
    "start2": 30120,
    "end2": 38120,
    "start3": 51520,
    "end3": 68520,
}

def plot_trajectories():
    HSV = np.load(scan_output_dir / "HSV.npy")
    H = HSV[:, 0]
    S = HSV[:, 1]
    V = HSV[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(1, 4):
        ax.scatter(H[results[f"start{i}"]: results[f"end{i}"]],
                S[results[f"start{i}"]: results[f"end{i}"]],
                V[results[f"start{i}"]: results[f"end{i}"]], 
                s=1, alpha=0.25)
        ax.scatter(H[results[f"start{i}"]],
                S[results[f"start{i}"]],
                V[results[f"start{i}"]], 
                s=100, c="r")
        ax.scatter(H[results[f"end{i}"]],
                S[results[f"end{i}"]],
                V[results[f"end{i}"]], 
                s=100, c="k")

    ax.scatter(H[results["start1"] - 2000],
            S[results["start1"] - 2000],
            V[results["start1"] - 2000], c="g")
    ax.scatter(H[results["start2"] - 2000],
            S[results["start2"] - 2000],
            V[results["start2"] - 2000], c="g")
    ax.scatter(H[results["start3"] - 2000],
            S[results["start3"] - 2000],
            V[results["start3"] - 2000], c="g")
    theta = np.linspace(0, 2*np.pi, 100)
    rs = 5
    x0, y0, z0 = 7.5, 20, 15
    x = np.linspace(-2.5, 2.5, len(theta)) + x0
    thetas, xs = np.meshgrid(theta, x)
    ys = rs*np.cos(thetas) + y0
    zs = rs*np.sin(thetas) + z0
    ax.plot_surface(xs, ys, zs)
    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(figure_dir / "3D manual.png")
    plt.show()

# plot_trajectories()