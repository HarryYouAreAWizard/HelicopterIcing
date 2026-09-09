

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
# data directory
mouse_movements_dir = Path(os.getcwd()) / "mouse_movements"
polygons_dir = Path(os.getcwd()) / "polygons"
figures_dir = Path(os.getcwd()) / "figures"

if not polygons_dir.is_dir():
    polygons_dir.mkdir(parents=True, exist_ok=True)


def load_polygon_corners(file):
    # run through the file and pick out when the left button is pressed
    xs, ys = [], []
    with open(mouse_movements_dir / file) as doc: 
        for line in doc:
            if "Button Button.left Pressed at" in line:
                x, y = line.split("(")[1].split(")")[0].split(",")
                xs.append(int(x))
                ys.append(int(y))
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

for f in os.listdir(mouse_movements_dir):
    if f[-2:] == "py":
        continue
    xs, ys = load_polygon_corners(f)
    np.save(polygons_dir / f"xs_{f[:-4]}.npy", xs)
    np.save(polygons_dir / f"ys_{f[:-4]}.npy", ys)



# xs_all, ys_all = load_polygon_corners(all)
# xs_yb, ys_yb = load_polygon_corners(yb)


# np.save(polygons_dir / "xs_all", xs_all)
# np.save(polygons_dir / "ys_all", ys_all)
# np.save(polygons_dir / "xs_yb", xs_yb)
# np.save(polygons_dir / "ys_yb", ys_yb)

# fig, ax=plt.subplots()
# ax.plot(xs_all - np.min(xs_all), -(ys_all - np.min(ys_all)), label="Entire apparatus")
# ax.plot(xs_yb - np.min(xs_yb), -(ys_yb - np.min(ys_yb)), label="Only yellow-black")
# ax.legend()
# fig.tight_layout()
# fig.savefig(figures_dir / "Apparatus polygons")
# plt.close()
