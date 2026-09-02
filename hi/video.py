

import numpy as np
import cv2

# wrapper functions
def set_frame(videocapture:cv2.VideoCapture, target_frame_index:int)->None:
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index) # search up the frame


def change_color_representation(frame:np.ndarray, command:int)->np.ndarray:
    """
    example command: cv2.COLOR_BGR2GRAY
    """
    cv2.COLOR_BAYER_BG2BGR
    representation = cv2.cvtColor(frame, command)
    return representation

def get_num_frames(videocapture:cv2.VideoCapture)->int:
    return int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))

# utility

def filter_frame(frame:np.ndarray, mask:np.ndarray)->np.ndarray:
    # filter the frame and normalize it
    return frame * mask[:, :, None]

def extract_pixels_in_mask(frame:np.ndarray, mask:np.ndarray)->np.ndarray:
    # pick out the valid pixels
    return frame[mask == 1]

def normalize_brightness(frame:np.ndarray, mask:np.ndarray)->np.ndarray:
    # normalization using LAB frame
    # https://en.wikipedia.org/wiki/CIELAB_color_space
    # L in LAB is "lightness"
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_frame)

    # extract the valid l-channel pixels
    valid_l_pixels = l_channel[mask==1].astype(np.float32)

    # normalize brightness channel by mean=255//2, std=50?
    l_mean, l_std = valid_l_pixels.mean(), valid_l_pixels.std()

    normalized_l = (valid_l_pixels - l_mean) / l_std * 50 + 128
    normalized_l = np.clip(normalized_l, 0, 255).astype(np.uint8)
    l_channel[mask==1] = normalized_l

    # merge back
    normalized_LAB = cv2.merge([l_channel, a_channel, b_channel])
    normalized = cv2.cvtColor(normalized_LAB, cv2.COLOR_LAB2BGR)
    return normalized


# show video
def load_and_show(videocapture:cv2.VideoCapture)->None:
    total_frames = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
    window_name = "flight"

    cv2.namedWindow(window_name)

    # method for setting frame
    def on_trackbar_change(trackbar_value):
        videocapture.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
        return 

    cv2.createTrackbar("...", window_name, 0, total_frames - 1, on_trackbar_change)

    while videocapture.isOpened():

        current_trackbar_pos = cv2.getTrackbarPos("...", window_name)

        ret, frame = videocapture.read()
    
        # if frame is read correctly ret is True. ret will be false when video is over
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # update the trackbar regularly
        current_frame_id = int(videocapture.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos("...", window_name, current_frame_id)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)#, gray)

        # actions to do
        # quit
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

    