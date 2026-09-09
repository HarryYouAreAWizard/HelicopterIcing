


import cv2
from video import get_num_frames, filter_frame, normalize_brightness, extract_pixels_in_mask
import numpy as np

from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score

def icenessindex(normalized_filtered_frame):

    # change representation
    HSV = cv2.cvtColor(normalized_filtered_frame, cv2.COLOR_BGR2HSV)

    H = HSV[:,:, 0].mean()
    S = HSV[:,:, 1].mean()
    V = HSV[:,:, 2].mean()
    x = np.array([H, S, V])

    if S+V < 1e-9:
        iceness = 10.0
    else:
        iceness = 1/(S + V)
    return iceness, x
    # isice = iceness>threshold

    # return iceness, isice, H, S, V


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
                videocapture.grab()

        if end_idx is not None and i == end_idx:
            break

    return HSV, RGB


def silhouette_score_from_mean_HSVs(X, iceness, thresholds, skip=1000):
    """
    Load saved HSV values for a video and obtain an array of silhouette scores from an array of thresholds
    """

    silhouette_scores = np.zeros(len(thresholds))
    for i in range(len(thresholds)):

        print(f"{i} / {thresholds.shape[0]}")
        labels = iceness > thresholds[i]

        # guard against bad thresholds
        if len(set(labels)) == 1:
            silhouette_scores[i] = np.nan
            continue
        # else:
            # print("Running silhouette score")            
        silhouette_scores[i] = silhouette_score(X, labels)

    return silhouette_scores


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




# get indicies for each period
def get_ice_indices(videocapture:cv2.VideoCapture, mask:np.ndarray, end_idx=None)->tuple:
    """
    decrepreated
    """
    # open video at beginning 
    i = 0
        
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, i) # search up the frame
    num_frames = get_num_frames(videocapture)

    if end_idx is None:
        end_idx = num_frames
        
    frame_indicies_ice = []
    frame_indicies_noice = []

    num_dp = 1
    while videocapture.isOpened():

        ret, frame = videocapture.read()
        if not ret:
            break

        frame = filter_frame(frame, mask)
        frame = normalize_brightness(frame, mask)

        _,isice,_,_,_ = icenessindex(frame)

        if isice:
            frame_indicies_ice.append(i)

        else:
            frame_indicies_noice.append(i)


        print(f"{i} / {num_frames}", end="\r", flush=True)

        i+=1
        if i == end_idx:
            break

    return frame_indicies_ice, frame_indicies_noice

