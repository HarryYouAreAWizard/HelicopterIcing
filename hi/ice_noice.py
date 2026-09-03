


import cv2
from video import get_num_frames, filter_frame, normalize_brightness
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


def scan_video(videocapture:cv2.VideoCapture, frame_mask:np.ndarray, thresholds:np.float64, 
               end_idx=None, debug=False, skip=None, print_interval=100)->np.ndarray:
    """
    decrepreated in favor of the scan video once - analyse afterwards method
    """

    n_frames = get_num_frames(videocapture)
    if end_idx is None:
        end_idx = n_frames - 1
    ice_mask = np.empty((end_idx, len(thresholds)))
    X = np.empty((end_idx, 3))
    i = 0
    while videocapture.isOpened():

        if i == end_idx:
            break

        ret, frame = videocapture.read()
        if not ret: break

        frame = filter_frame(frame, frame_mask)
        frame = normalize_brightness(frame, frame_mask)

        iceness, x = icenessindex(frame)

        ice_mask[i, :] = iceness > thresholds
        X[i, :] = x


        i+=1
        if i%print_interval==0:
            print(f"{i} / {end_idx}")

        if skip is None:
            continue

        for _ in range(skip):
            videocapture.grab()
        
    # artificial entries for debugging
    if debug:
        ice_mask[0, :] = 0
        ice_mask[1, :] = 1

    return ice_mask, X

def scan_thresholds(videocapture:cv2.VideoCapture, frame_mask:np.ndarray, thresholds:np.ndarray, 
                    rescan_video=False, intermediate_storage=None, end_idx=None, debug=False, skip=None, print_interval=100)->np.ndarray:
    """
    decrepreated in favor of the scan video once - analyse afterwards method
    """

    ice_mask, X = scan_video(videocapture, frame_mask, thresholds, end_idx=end_idx, debug=debug, skip=skip, print_interval=print_interval)
    silhouette_scores = np.empty_like(thresholds)

    for i, _ in enumerate(thresholds):
        # let the ice_mask be the labels instead
        # _,labels,_ = k_means(X=ice_mask[:, i:i+1], n_clusters=2)

        # labels = ice_mask[:, i:i+1]
        labels = ice_mask[:, i]
        # guard agains bad thresholds
        if len(set(labels)) == 1:
            silhouette_scores[i] = np.nan
            continue

        silhouette_scores[i] = silhouette_score(X, labels)
    return silhouette_scores, X, ice_mask


def get_video_frame_averaged_HSV(videocapture:cv2.VideoCapture, frame_mask:np.ndarray, end_idx=None, skip=None):
    """the slowest part of the icing index analysis is getting the HSV values.

    Here we scan the entire video and the the average HSV for every frame
     
    """
    n_frames = get_num_frames(videocapture)
    X = np.empty((n_frames, 3))
    i = 0
    while videocapture.isOpened():

        ret, frame = videocapture.read()
        if not ret: break

        frame = filter_frame(frame, frame_mask)
        frame = normalize_brightness(frame, frame_mask)

        _, x = icenessindex(frame)

        X[i, :] = x


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

    return X


def silhouette_score_from_mean_HSVs(HSV, thresholds, skip=1000):
    """
    Load saved HSV values for a video and obtain an array of silhouette scores from an array of thresholds
    """

    HSV = HSV[::skip, :]

    # H = HSV[:, 0]
    S = HSV[:, 1]
    V = HSV[:, 2]

    iceness = 1/(S+V) 

    silhouette_scores = np.zeros(len(thresholds))
    for i in range(len(thresholds)):

        print(f"{i} / {thresholds.shape[0]}")
        labels = iceness > thresholds[i]

        # guard against bad thresholds
        if len(set(labels)) == 1:
            silhouette_scores[i] = np.nan
            continue
        else:
            print("Running silhouette score")            
        silhouette_scores[i] = silhouette_score(HSV, labels)

    return silhouette_scores



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

