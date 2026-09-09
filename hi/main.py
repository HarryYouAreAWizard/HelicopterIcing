



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

figure_dir          = Path(os.getcwd()) / "figures"
data_dir            = Path(os.getcwd()) / "video-data"
polygon_dir         = Path(os.getcwd()) / "polygons"
case_videos_dir     = Path(os.getcwd()) / "case_videos"
segnet_output_dir   = Path(os.getcwd()) / "segnet_output"
scan_output_dir     = Path(os.getcwd()) / "scan_output"
reduced_video_dir   = Path(os.getcwd()) / "reduced_video_data"


video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"

# to avoid spam, even though the video works fine
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"



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

    cap = cv2.VideoCapture(data_dir / video_filename)
    pipe_mask = mask_creation.get_mask(cap, "pipe_small", plot_mask=True, figure_dir=figure_dir)

   
    start = 10000
    end = 40000
    i = start
    set_frame(cap, i)
    under_history = []
    above_history = []
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Original", frame)
        frame = filter_frame(frame, pipe_mask)
        gray = change_color_representation(frame, cv2.COLOR_BGR2GRAY)


        nonzero = cv2.findNonZero(pipe_mask)

        x_min = nonzero[:, 0].min()
        x_max = nonzero[:, 0].max()
        y_min = nonzero[:, 1].min()
        y_max = nonzero[:, 1].max()

        # cv2.drawMarker(gray, (x_min, y_min), 100)
        # cv2.drawMarker(gray, (x_max, y_max), 100)


        relevant_frame = gray[y_min:y_max, x_min:x_max]
        cv2.imshow("Gray pipe", relevant_frame)

        some_point = (30, 10)

        # cv2.imwrite(figure_dir / "image.png", relevant_frame)


        # Detect horizontal edges (Sobel X)
        # sobel = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=3, dy=3, ksize=7)

        # # Detect vertical edges (Sobel Y)
        # sobely = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        # edge_x = cv2.convertScaleAbs(sobelx)
        # edges = cv2.convertScaleAbs(sobel)
        print(f"{i}", end="\r")
        
        # cv2.imshow("Edges", edges)
        # cv2.imshow("Edges y", edge_y)

        # n_under = under_mask[i//10,:].sum()
        # n_above = above_mask[i//10,:].sum()
        # under_history.append(n_under)
        # above_history.append(n_above)
        # fig, ax=plt.subplots()
        # ax.plot(under_history)
        # ax.plot(above_history)
        # fig.savefig(figure_dir / "history.png")
        # # print(f"{n_under = }")
        # # print(f"{n_above = }")
        # plt.close("all")

        i += 1

        key = cv2.waitKey(1)

        # quit
        if key == ord("q"):
            break

        # pause
        elif key == ord(" "):
            new_key = cv2.waitKey(0)
            if new_key == ord(" "):
                continue
        elif key == ord("s"):
            i = int(input("new frame: "))
            set_frame(cap, i)


        if i == end:
            break

    cv2.destroyAllWindows()



    return 
    cap_mask = mask_creation.get_mask(cap, "yellow_black", plot_mask=True, figure_dir=figure_dir)
    pipe_mask = mask_creation.get_mask(cap, "pipe", plot_mask=True, figure_dir=figure_dir)

    # HSV, _ = get_video_frame_FULL_HSV(cap, cap_mask, False, skip=1)
    # np.save(scan_output_dir / "HSV_pipe_cap_light.npy", HSV[::10])
    # print(f"{HSV.shape = }")
    
    # try:
    #     HSV = np.load(scan_output_dir / "HSV.npy")
    #     RGB = np.load(scan_output_dir / "RGB.npy")
    # except Exception:
    HSV, RGB = get_video_frame_averaged_HSV(cap, cap_mask, normalize=False)
    np.save(scan_output_dir / "HSV.npy", HSV)
    np.save(scan_output_dir / "RGB.npy", RGB)

    skip = 10
    center = 50
    half_width = center - 0.05
    thresholds = np.linspace(center-half_width, center+half_width, 10000)
    np.save(scan_output_dir / "thresholds.npy", thresholds)
    HSV = HSV[::skip, :]
    RGB = RGB[::skip, :]

    H = HSV[:, 0]
    S = HSV[:, 1]
    V = HSV[:, 2]
    R = RGB[:, 0]
    G = RGB[:, 1]
    B = RGB[:, 2]

    HSV_variations = {
        "HSV": H+S+V,
        "HS": H+S,
        "HV": H+V,
        "SV": S+V,
        "H": H,
        "S": S,
        "V": V,
    }
    RGB_variations = {
        "RGB": R+G+B,
        "RG": R+G,
        "RB": R+B,
        "GB": G+B,
        "R": R,
        "G": G,
        "B": B,
    }

    for variation in HSV_variations.keys():
        iceness = HSV_variations[variation]
        S = silhouette_score_from_mean_HSVs(HSV, iceness, thresholds, skip=10)
        np.save(scan_output_dir / f"silhouette score {variation}.npy", S)
    
    for variation in RGB_variations.keys():
        iceness = RGB_variations[variation]
        S = silhouette_score_from_mean_HSVs(RGB, iceness, thresholds, skip=10)
        np.save(scan_output_dir / f"silhouette score {variation}.npy", S)
    

    # print("Finding silhouette scores...")
    # thresholds, silhouette_scores = run_silhouette_score(skip=5)

    # fig, ax=plt.subplots()
    # # ax.plot(thresholds, silhouette_scores)
    # ax.set_xlabel("Threshold")
    # ax.set_ylabel("Silhouette score")
    # ax.set_title("Optimizing icing threshold")
    # fig.tight_layout()
    # fig.savefig(figure_dir / "thresholds-silhouette.png")





    # num_bins = 50
    # while red.isOpened():
    #     fig, axs=plt.subplots(2, 1) 
    #     ret, BGR = red.read()
    #     if not ret: break
    #     BGR = filter_frame(BGR, red_mask)

    #     LAB = change_color_representation(BGR, cv2.COLOR_BGR2LAB)
    #     pixels = extract_pixels_in_mask(LAB, red_mask)[:, 0]
    #     counts, bins = np.histogram(pixels, bins=num_bins)
    #     axs[0].plot(bins[:-1], counts)

    #     BGR = normalize_brightness(BGR, red_mask)

    #     LAB = change_color_representation(BGR, cv2.COLOR_BGR2LAB)
    #     pixels = extract_pixels_in_mask(LAB, red_mask)[:, 0]
    #     counts, bins = np.histogram(pixels, bins=num_bins)
    #     axs[1].plot(bins[:-1], counts)

    #     fig.savefig(figure_dir / "histogram.png")

    #     cv2.imshow(".", BGR)
    #     plt.close()

    #     key = cv2.waitKey(10)
    #     if key == ord("q"):
    #         break
    #     if key == ord("s"):
    #         i = input("frame index: ")
    #         i = int(i)
    #         set_frame(red, i)

    # cv2.destroyAllWindows()
main()