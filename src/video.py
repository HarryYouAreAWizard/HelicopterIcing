

import numpy as np
import cv2

# wrapper functions
def set_frame(videocapture, target_frame_index):
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index) # search up the frame


def change_color_representation(frame, command):
    """
    example command: cv2.COLOR_BGR2GRAY
    """
    representation = cv2.cvtColor(frame, command)
    return representation

# utility
def normalize_brightness(frame, mask):
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