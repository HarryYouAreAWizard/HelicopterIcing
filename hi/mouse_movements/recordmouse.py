
import cv2
import os
from pynput import mouse
from pathlib import Path

data_dir = Path("..") / "data"
figure_dir = Path("figures")
video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"


def on_move(x, y):
    # print(f"Mouse moved to position: ({x}, {y})")
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

cap = cv2.VideoCapture(data_dir / video_filename)
ret, frame = cap.read()
print("Running mouse listener...")
with mouse.Listener(
        on_move=on_move, 
        on_click=on_click, 
        on_scroll=on_scroll
    ) as listener: 
    print(f"{listener.is_alive() = }")

    # set to first frame
    # cv2.imwrite(figure_dir / "image.png", frame)
    cv2.imshow("..", frame)
    cv2.waitKey(0)
    listener.join()
cap.release()

# print(f"\n"*100)
