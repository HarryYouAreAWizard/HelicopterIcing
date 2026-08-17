

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

mouse_movements_folder = Path("..") / "mouse_movements"

all = "all_apparatus.txt"
yb = "yellow_black.txt"

def load_polygon_corners(file):
    # run through the file and pick out when the left button is pressed
    xs, ys = [], []
    with open(mouse_movements_folder / file) as doc: 
        for line in doc:
            if "Button Button.left Pressed at" in line:
                x, y = line.split("(")[1].split(")")[0].split(",")
                xs.append(int(x))
                ys.append(int(y))
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

xs_all, ys_all = load_polygon_corners(all)
xs_yb, ys_yb = load_polygon_corners(yb)

fig, ax=plt.subplots()
ax.plot(xs_all - np.min(xs_all), -(ys_all - np.min(ys_all)), label="Entire apparatus")
ax.plot(xs_yb - np.min(xs_yb), -(ys_yb - np.min(ys_yb)), label="Only yellow-black")
ax.legend()
fig.tight_layout()
fig.savefig(Path("..") / "figures" / "Apparatus polygons")
plt.close()
