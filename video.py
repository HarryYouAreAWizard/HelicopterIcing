


import cv2

def set_frame(videocapture, target_frame_index):
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index) # search up the frame


def change_color_representation(frame, command):
    """
    example command: cv2.COLOR_BGR2GRAY
    """
    representation = cv2.cvtColor(frame, command)
    return representation