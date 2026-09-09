
# import cv2
# import os
from pynput import mouse
# from pathlib import Path
# from video import set_frame

# # data_dir = Path(os.getcwd()) / "video-data"
# # figure_dir = Path(os.getcwd()) / "figures"
# # video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"


def on_move(x, y):
    print(f"Mouse moved to position: ({x}, {y})")
    pass

def on_click(x, y, button, pressed):
    action = "Pressed" if pressed else "Released"
    print(f"Button {button} {action} at ({x}, {y})")

    # stop on right button 
    if button == mouse.Button.right and not pressed:
        print("Stopping listener...")
        return False

def on_scroll(x, y, dx, dy):
    pass
    # direction = "Down/Right" if (dx < 0 or dy < 0) else "Up/Left"
    # print(f"Scrolled {direction} at ({x}, {y})")

# set to first frame
# cap = cv2.VideoCapture(data_dir / video_filename)
# ret, frame = cap.read()
# cv2.imwrite(figure_dir / "image.png", frame)
# cap.release()

# print(f"\n"*100)
print("Running mouse listener...")
# with mouse.Listener(
#         on_move=on_move, 
#         on_click=on_click, 
#         on_scroll=on_scroll
#     ) as listener: 
#     print(f"{listener.is_alive() = }")
#     listener.join()
listener = mouse.Listener(
        on_move=on_move, 
        on_click=on_click, 
        on_scroll=on_scroll
    )
listener.start()
print(f"joining")
listener.join()

