


import numpy as np
import cv2
from convmodel import MaskedVideo
import torch


def get_pipe_params(frame:np.ndarray)->dict:

    xmin = 0
    ymin = 0
    xmax = frame.shape[1] - 1
    ymax = frame.shape[0] - 1

    list_of_xs_ys_where_nonzero = cv2.findNonZero(frame)

    x_left,   y_left    = list_of_xs_ys_where_nonzero[list_of_xs_ys_where_nonzero[:, 0] == xmin][-1]
    x_top,    y_top     = list_of_xs_ys_where_nonzero[list_of_xs_ys_where_nonzero[:, 1] == ymin][-1]
    x_right,  y_right   = list_of_xs_ys_where_nonzero[list_of_xs_ys_where_nonzero[:, 0] == xmax][-1]
    x_bottom, y_bottom  = list_of_xs_ys_where_nonzero[list_of_xs_ys_where_nonzero[:, 1] == ymax][-1]
    print(f"left:   {x_left,   y_left}")
    print(f"top:    {x_top,    y_top}")
    print(f"right:  {x_right,  y_right}")
    print(f"bottom: {x_bottom, y_bottom}")

    def line(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        slope = (y2 - y1) / (x2 - x1)
        bias = y2 - slope * x2
        return slope, bias


    slope_base,   bias_base   = line((x_right, y_right), (x_bottom, y_bottom))
    slope_farend, bias_farend = line((x_left,  y_left),  (x_top,    y_top))
    slope_upper,  bias_upper  = line((x_right, y_right), (x_top,    y_top))
    slope_lower,  bias_lower  = line((x_left,  y_left),  (x_bottom, y_bottom))


    result_dict = {
        "point": {
            "left":   (x_left,   y_left),
            "top":    (x_top,    y_top),
            "right":  (x_right,  y_right),
            "bottom": (x_bottom, y_bottom),
        },
        "line": {
            "base": (slope_base,   bias_base),
            "farend": (slope_farend, bias_farend),
            "upper": (slope_upper,  bias_upper),
            "lower": (slope_lower,  bias_lower),
        }
    }
    return result_dict




# calculating intersections
# https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b
# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


def calculate_intersections(pipe_corners, prediction_points):
    prediction_point1, prediction_point2 = prediction_points 
    intersections = {}
    for name, (slope, bias) in pipe_corners["line"].items():
        x1, x2 = pipe_corners["point"]["left"][0], pipe_corners["point"]["right"][0]
        y1 = np.int32(slope * x1 + bias) 
        y2 = np.int32(slope * x2 + bias) # problem found! Conversion to uint8. Either overflow or ignoreing negative values? 

        pred_x1, pred_y1 = prediction_point1
        pred_x2, pred_y2 = prediction_point2

        intersection = seg_intersect(
            np.array((x1, y1)), 
            np.array((x2, y2)), 
            np.array(prediction_point1),
            np.array(prediction_point2)
        ).astype(np.int32)

        intersections[name] = intersection
    return intersections

def calculate_pixel_apparent_thickness(pipe_corners, prediction):

    slope, bias = prediction 
    prediction_y1 = np.int32(slope * pipe_corners["point"]["bottom"][0] + bias)
    prediction_y2 = np.int32(slope * pipe_corners["point"]["right"][0] + bias)
    prediction_point1 = np.array((pipe_corners["point"]["bottom"][0], prediction_y1))
    prediction_point2 = np.array((pipe_corners["point"]["right"][0], prediction_y2))

    intersections = calculate_intersections(
        pipe_corners, 
        prediction_points=(prediction_point1, prediction_point2)
    )

    corner1 = np.array(pipe_corners["point"]["right"])
    corner2 = np.array(pipe_corners["point"]["bottom"])
    intersection1 = intersections["upper"]
    intersection2 = intersections["lower"]
    d1 = np.linalg.norm(intersection1 - corner1)
    d2 = np.linalg.norm(intersection2 - corner2)

    apparent_pixel_thickness = np.mean([d1, d2])
    return apparent_pixel_thickness, intersections


def run_apparent_thickness(videocapture, model):
    data = MaskedVideo(videocapture)
    model.eval()


    frame = data[0]
    # frame.shape = [1, 48, 52] = [C, H, W] -> [H, W]
    frame = frame[0, :, :].numpy()
    pipe_corners = get_pipe_params(frame)
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    apparent_thicknesses = np.empty(len(data))
    for i, batch in enumerate(dataloader):
        predictions = model.predict(batch)

        for j, prediction in enumerate(predictions):
            thickness, _ = calculate_pixel_apparent_thickness(pipe_corners, prediction.numpy())
            apparent_thicknesses[i*batch_size + j] = thickness

        print(f"finished {i+1} / {len(data)//batch_size}")

    return apparent_thicknesses