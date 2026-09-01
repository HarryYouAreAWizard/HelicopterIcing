


import cv2
import video


def icenessindex(normalized_filtered_frame):
    # H_lower, H_upper = 0, 3
    # S_lower, S_upper = 0, 3.0  
    # V_lower, V_upper = 0, 7.75
    # isice = (H < H_upper and S < S_upper and V < V_upper)

    ICE_threshold = 0.095
    # change representation
    HSV = cv2.cvtColor(normalized_filtered_frame, cv2.COLOR_BGR2HSV)

    H = HSV[:,:, 0].mean()
    S = HSV[:,:, 1].mean()
    V = HSV[:,:, 2].mean()

    # iceness = 1/(H + S + V)
    iceness = 1/(S + V)
    isice = iceness>ICE_threshold
    # return isice, iceness
    return iceness, isice, H, S, V


# get indicies for each period
def get_ice_indices(videocapture:cv2.VideoCapture, mask)->tuple:
    # open video at beginning 
    i = 0
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, i) # search up the frame
    num_frames = video.get_num_frames(videocapture)
    ret, frame = videocapture.read()

    frame_indicies_ice = []
    frame_indicies_noice = []

    num_dp = 1
    while videocapture.isOpened():

        ret, frame = videocapture.read()
        if not ret:
            break

        frame = video.filter_frame(frame, mask)
        frame = video.normalize_brightness(frame, mask)

        _,isice,_,_,_ = icenessindex(frame)

        if isice:
            frame_indicies_ice.append(i)

        else:
            frame_indicies_noice.append(i)

        i+=1

        print(f"{i} / {num_frames}", end="\r", flush=True)
    return frame_indicies_ice, frame_indicies_noice

