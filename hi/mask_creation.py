



import os
import numpy as np
import cv2
from pathlib import Path

# polygon_dir = Path("..") / "polygons"
polygon_dir = Path("polygons")
# polygon_dir = Path(__file__[:len(__name__)+2]) / "polygons"

# define method for determining whether a point is inside a polygon
# https://www.geeksforgeeks.org/dsa/how-to-check-if-a-given-point-lies-inside-a-polygon/
def on_segment(x1, y1, x2, y2, x, y):
    c = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    if c != 0:
        return False

    return (min(x1, x2) <= x <= max(x1, x2) and
            min(y1, y2) <= y <= max(y1, y2))

def isInside(arr, x, y):
    n = len(arr)
    inside = False

    j = n - 1

    for i in range(n):
        x1, y1 = arr[i]
        x2, y2 = arr[j]

        # Point lies on the current edge
        if on_segment(x1, y1, x2, y2, x, y):
            return True

        # Check whether the horizontal ray from (x, y)
        # intersects the current edge
        intersect = ((y1 > y) != (y2 > y) and
                     x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)

        if intersect:
            inside = not inside

        j = i

    return inside

# fit the polygons to match position of wing in frame
# made manually
def fit_polygon_all_wing(xs, ys, frame_shape):
    # shift polygons to upper right corner
    xs -= np.min(xs)
    ys -= np.min(ys)

    # move to proper startting position
    x_move = frame_shape[1]//2 - 10
    y_move = frame_shape[0]//2 + 85
    xs += x_move 
    ys += y_move 
    return xs, ys

def fit_polygon_yellow_black_wing(xs, ys, frame_shape):
    # shift polygons to upper right corner
    xs -= np.min(xs)
    ys -= np.min(ys)

    # move to proper startting position
    x_move = frame_shape[1]//2 + 100
    y_move = frame_shape[0]//2 + 85
    xs += x_move 
    ys += y_move 
    return xs, ys


def draw_polygon(polygon, frame):
    previous_point = polygon[0]
    for point in polygon[1:]:
        cv2.line(frame, previous_point, point, (255, 0, 0), 5)
        previous_point = point
    cv2.line(frame, polygon[-1], polygon[0], (255, 0, 0), 5)

def load_fitted_polygon(frame_shape, which_polygon="yb"):
    # from mask_creation import fit_polygon_all_wing, fit_polygon_yellow_black_wing
    # load polygons
    xs_all = np.load(polygon_dir / "xs_all.npy")
    ys_all = np.load(polygon_dir / "ys_all.npy")
    xs_yb = np.load(polygon_dir / "xs_yb.npy")
    ys_yb = np.load(polygon_dir / "ys_yb.npy")

    # fit the polygons to the frame
    xs_all, ys_all = fit_polygon_all_wing(xs_all, ys_all, frame_shape)
    xs_yb, ys_yb = fit_polygon_yellow_black_wing(xs_yb, ys_yb, frame_shape)

    polygons = {
        "yb": list(zip(xs_yb, ys_yb)),
        "all": list(zip(xs_all, ys_all))
    }

    return polygons[which_polygon]

def create_mask(polygon, frame_shape):
    # create a mask using the polygon
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(
        mask, 
        [np.asarray(polygon, dtype=np.int32)], 
        1
    )
    return mask


def reduced_frame_polygon(frame):
    polygon = load_fitted_polygon(frame.shape)
    polygon = np.array(polygon)
    polygon[:, 0] -= 325
    polygon[:, 1] -= 190
    return polygon

def get_mask(videocapture, plot_mask=False, figure_dir=None, which_polygon="yb"):

    # set first frame and get the shape
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    ret, first_frame = videocapture.read()
    if not ret:
        print(f"Error during videocapture read")
        return 0
    frame_shape = first_frame.shape

    # load the fitted polygon
    polygon = load_fitted_polygon(frame_shape, which_polygon=which_polygon)

    # draw the polygon on the first frame. Visualized as to help fitting the polygon
    draw_polygon(polygon, first_frame)

    mask = create_mask(polygon, frame_shape)

    if plot_mask:
        # plot the masked image along with the drawing of the polygon
        first_frame_masked = first_frame * mask[:, :, None]
        # cv2.imwrite(Path("..") /"figures" / "image.png", first_frame_masked)
        cv2.imwrite(figure_dir / "mask_example.png", first_frame_masked)

    return mask
