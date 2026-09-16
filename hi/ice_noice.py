


import cv2
from video import get_num_frames, filter_frame, normalize_brightness, extract_pixels_in_mask
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score
import mask_creation

figure_dir = Path(os.getcwd()) / "figures"
scan_output_dir = Path(os.getcwd()) / "scan_output"

# -----------------Video scans-----------------
def get_video_frame_averaged_HSV(videocapture:cv2.VideoCapture, frame_mask:np.ndarray, normalize=False, end_idx=None, skip=None):
    """the slowest part of the icing index analysis is getting the HSV values.

    Here we scan the entire video and the the average HSV for every frame
     
    """
    n_frames = get_num_frames(videocapture)
    HSV = np.empty((n_frames, 3))
    RGB = np.empty_like(HSV)

    i = 0
    while videocapture.isOpened():

        ret, frame = videocapture.read()
        if not ret: break

        frame = filter_frame(frame, frame_mask)
        
        if normalize:
            frame = normalize_brightness(frame, frame_mask)

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        r = rgb[:,:, 0].mean()
        g = rgb[:,:, 1].mean()
        b = rgb[:,:, 2].mean()
        h = hsv[:,:, 0].mean()
        s = hsv[:,:, 1].mean()
        v = hsv[:,:, 2].mean()

        rgb = np.array([r, g, b])
        hsv = np.array([h, s, v])

        # _, hsv = icenessindex(frame)

        HSV[i, :] = hsv
        RGB[i, :] = rgb


        if i == n_frames - 1:
            break

        i += 1

        if i%1000 == 0:
            print(f"{i} / {n_frames}")

        if skip is not None:

            for _ in range(skip):
                break # logic is wrong, indices will be wrong
                videocapture.grab()

        if end_idx is not None and i == end_idx:
            break

    return HSV, RGB

def get_video_frame_FULL_HSV(videocapture:cv2.VideoCapture, frame_mask:np.ndarray, normalize=False, end_idx=None, skip=None):
    """
     
    """
    n_frames = get_num_frames(videocapture) // skip
    n_pixels = frame_mask.sum() // skip
    HSV = np.empty((n_frames, n_pixels, 3))
    RGB = np.empty_like(HSV)

    i = 0
    while videocapture.isOpened():

        ret, frame = videocapture.read()
        if not ret: break

        frame = filter_frame(frame, frame_mask)
        
        if normalize:
            frame = normalize_brightness(frame, frame_mask)

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        rgb = extract_pixels_in_mask(rgb, frame_mask)
        hsv = extract_pixels_in_mask(hsv, frame_mask)

        HSV[i, :, :] = hsv[::skip]#[:-1]
        RGB[i, :, :] = rgb[::skip]#[:-1]


        if i == n_frames - 1:
            break

        i += 1

        if i%100 == 0:
            print(f"{i} / {n_frames}")

        if skip is not None and skip > 1:
            for _ in range(skip-1):
                videocapture.grab()

        if end_idx is not None and i == end_idx:
            break

    return HSV, RGB

def run_mean_extraction(cap, included_area):
    mask = mask_creation.get_mask(cap, included_area, plot_mask=True, figure_dir=figure_dir)

    HSV, RGB = get_video_frame_averaged_HSV(cap, mask, normalize=False)
    np.save(scan_output_dir / f"HSV_{included_area}.npy", HSV)
    np.save(scan_output_dir / f"RGB_{included_area}.npy", RGB)


# -----------------Silhouette calculations-----------------
def silhouette_score_from_mean_HSVs(X, iceness, thresholds):
    """
    Load saved HSV values for a video and obtain an array of silhouette scores from an array of thresholds
    """

    silhouette_scores = np.zeros(len(thresholds))
    for i in range(len(thresholds)):

        print(f"{i} / {thresholds.shape[0]}", end="\r")
        labels = iceness > thresholds[i]

        # guard against bad thresholds
        if len(set(labels)) == 1:
            silhouette_scores[i] = np.nan
            continue
        # else:
            # print("Running silhouette score")            
        silhouette_scores[i] = silhouette_score(X, labels)

    return silhouette_scores

def remove_nans(thresholds, silhouette_scores):
    """
    some silhouette scores cannot be calculated, since all points are placed in the same cluster
    
    this function is used before plotting"""
    t_ = []
    s_ = []
    for ts, ss in zip(thresholds, silhouette_scores):
        if not np.isnan(ss):
            t_.append(ts)
            s_.append(ss)
    return np.array(t_), np.array(s_)


def run_silhouette_score(included_area, skip=10, n_thresholds=10000):
    center = 50
    half_width = center - 0.001

    thresholds = np.linspace(center-half_width, center+half_width, n_thresholds)
    np.save(scan_output_dir / f"thresholds_{included_area}.npy", thresholds)
    # HSV = HSV[::skip, :]
    # RGB = RGB[::skip, :]
    HSV = np.load(scan_output_dir / f"HSV_{included_area}.npy")[::skip, :]
    RGB = np.load(scan_output_dir / f"RGB_{included_area}.npy")[::skip, :]


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
        S = silhouette_score_from_mean_HSVs(HSV, iceness, thresholds)
        np.save(scan_output_dir / f"silhouette_score_{included_area}_{variation}.npy", S)
    
    for variation in RGB_variations.keys():
        iceness = RGB_variations[variation]
        S = silhouette_score_from_mean_HSVs(RGB, iceness, thresholds)
        np.save(scan_output_dir / f"silhouette_score_{included_area}_{variation}.npy", S)

# -----------------Results get and show-----------------
def pixel_rep_plot(variations, variation, optimal_thresholds, included_area):
    """show how every pixel is clustered"""

    # get clustering
    labels = variations[variation] > optimal_thresholds[variation]

    # Pick out the variation with the full representation
    title = list(variations.keys())[0]
    print(f"{title = }")

    if title=="HSV":
        HSV_RGB = np.load(scan_output_dir / f"HSV_{included_area}.npy")
    if title=="RGB":
        HSV_RGB = np.load(scan_output_dir / f"RGB_{included_area}.npy")

    print(f"{HSV_RGB.shape = }")
    H_R = HSV_RGB[:, 0]
    S_G = HSV_RGB[:, 1]
    V_B = HSV_RGB[:, 2]
    fig, axs=plt.subplots(1, 3)
    axs[0].scatter(H_R[labels==0], S_G[labels==0], c="tab:blue", label="0")
    axs[0].scatter(H_R[labels==1], S_G[labels==1], c="tab:orange", label="1")
    axs[1].scatter(H_R[labels==0], V_B[labels==0], c="tab:blue", label="0")
    axs[1].scatter(H_R[labels==1], V_B[labels==1], c="tab:orange", label="1")
    axs[2].scatter(S_G[labels==0], V_B[labels==0], c="tab:blue", label="0")
    axs[2].scatter(S_G[labels==1], V_B[labels==1], c="tab:orange", label="1")
    axs[0].set_xlabel(title[0]) # "H"
    axs[0].set_ylabel(title[1]) # "S"
    axs[1].set_xlabel(title[0]) # "H"
    axs[1].set_ylabel(title[2]) # "V"
    axs[2].set_xlabel(title[1]) # "S"
    axs[2].set_ylabel(title[2]) # "V"
    for ax in axs: 
        ax.legend()
    fig.suptitle(f"Optimized w/ resp. to {variation}")
    fig.tight_layout()
    fig.savefig(figure_dir / f"Pixel_representation_plot_{included_area}_{variation}.png")
    plt.close("all")

def pixel_rep_plot_3D(variations, variation, optimal_thresholds, included_area):
    """show how every pixel is clustered"""

    # get clustering
    labels = variations[variation] > optimal_thresholds[variation]

    # Pick out the variation with the full representation
    title = list(variations.keys())[0]
    print(f"{title = }")

    if title=="HSV":
        HSV_RGB = np.load(scan_output_dir / f"HSV_{included_area}.npy")
    if title=="RGB":
        HSV_RGB = np.load(scan_output_dir / f"RGB_{included_area}.npy")

    print(f"{HSV_RGB.shape = }")
    H_R = HSV_RGB[:, 0]
    S_G = HSV_RGB[:, 1]
    V_B = HSV_RGB[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(H_R[labels==0],
               S_G[labels==0],
               V_B[labels==0], 
               s=1)
    ax.scatter(H_R[labels==1],
               S_G[labels==1],
               V_B[labels==1], 
               s=1)
    ax.set_xlabel(title[0])
    ax.set_ylabel(title[1])
    ax.set_zlabel(title[2])
    ax.set_aspect("equal")
    fig.suptitle(f"Optimized w/ resp. to {variation}")
    fig.tight_layout()
    fig.savefig(figure_dir / f"Pixel_representation_plot_3D_{included_area}_{variation}.png")
    plt.close("all")


def silhouette_optimized_threshold(thresholds, variations, included_area, make_plot=False):
    if make_plot: fig, axs=plt.subplots(1, 7, sharex=True, sharey=True, figsize=(15,8))
    optimal_thresholds = {}
    for i, variation in enumerate(variations.keys()):
        silhouette_scores = np.load(scan_output_dir / f"silhouette_score_{included_area}_{variation}.npy")
        
        t, s = remove_nans(thresholds, silhouette_scores)
        try:
            optimal_thresholds[variation] = t[s == s.max()][0]
            print(f" {variation} optimal_threshold: {t[s == s.max()][0]}")
            if make_plot:
                pixel_rep_plot_3D(variations, variation, optimal_thresholds, included_area)
        except ValueError:
            print(f"{included_area}, {variation} found no optimal threshold")


        if not make_plot: 
            continue
        ax = axs.flat[i]
        ax.plot(t, s)
        ax.set_title(variation)
        try:
            ax.scatter(t[s == s.max()], 
                    s[s == s.max()],
                    c="r")
        except Exception: pass
        plt.close("all")
        
        
    if make_plot: 
        fig.supxlabel("Threshold")
        fig.supylabel("Silhouette score")
        fig.tight_layout()
        fig.savefig(figure_dir / f"thresholds-silhouette_{list(variations.keys())[0]}_{included_area}.png")

    return optimal_thresholds


def get_optimal_thresholds(included_area, make_plot=True):

    thresholds = np.load(scan_output_dir / f"thresholds_{included_area}.npy")
    HSV = np.load(scan_output_dir / f"HSV_{included_area}.npy")
    RGB = np.load(scan_output_dir / f"RGB_{included_area}.npy")


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

    optimal_thresholds_HSV = silhouette_optimized_threshold(thresholds, HSV_variations, included_area=included_area, make_plot=make_plot)
    optimal_thresholds_RGB = silhouette_optimized_threshold(thresholds, RGB_variations, included_area=included_area, make_plot=make_plot)

    return optimal_thresholds_HSV, optimal_thresholds_RGB


# -----------------k-means clustering-----------------
def cluster_with_k_means(variation, included_area):
    # import algorithm from sklearn
    from sklearn.cluster import k_means

    # load the data
    HSV_RGB = np.load(scan_output_dir / f"{variation}_{included_area}.npy")
    X = HSV_RGB
    X = X.reshape((X.shape[0], -1))
    H_R = HSV_RGB[:, 0]
    S_G = HSV_RGB[:, 1]
    V_B = HSV_RGB[:, 2]

    # perform the clustering
    centroids, labels, _ = k_means(X, n_clusters=2)

    # manual 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(H_R[labels==0],
               S_G[labels==0],
               V_B[labels==0], 
               s=1, alpha=0.25)
    ax.scatter(H_R[labels==1],
               S_G[labels==1],
               V_B[labels==1], 
               s=1, alpha=0.25)
    
    print(f"{centroids = }")
    ax.scatter(*centroids[0], c="r", s=100)
    ax.scatter(*centroids[1], c="r", s=100)
    ax.set_xlabel(variation[0])
    ax.set_ylabel(variation[1])
    ax.set_zlabel(variation[2])
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.suptitle(f"k-means clustered, {included_area}, {variation}")
    fig.savefig(figure_dir / f"k_means_clustered_{included_area}_{variation}.png")