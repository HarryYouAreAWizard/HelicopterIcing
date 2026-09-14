



import torch
from torch import nn, tensor
import pandas
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import sleep
import mask_creation

data_dir  = Path(os.getcwd()) / "video-data"
label_dir = Path(os.getcwd()) / "label_data"
model_dir = Path(os.getcwd()) / "model_weights"
figure_dir = Path(os.getcwd()) / "figures"
video_filename = "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"


class Linefinder(nn.Module):

    def __init__(self, frame_shape = (1, 48, 52)):
        super().__init__()

        

        channels, height, width = frame_shape
        filter_size = 5
        out_channels = 2
        self.pooling_kernel_size = 4

        self.features = nn.Sequential(
            # [B, 1, 48, 52] -> [B, 16, 48, 52]
            nn.Conv2d(channels, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),   # similar to batch size, but normalizes over groups of channels instead of the batch
            nn.SiLU(),  # x*sigmoid(x)

            # -> [B, 16, 24, 26]
            nn.MaxPool2d(kernel_size=2),

            # -> [B, 32, 24, 26]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d((3, 4)), # always outputs the same shape, regardless of input. Nice if we change the pipe mask later
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),              # 64 * 3 * 4 = 768 features
            nn.Linear(32 * 3 * 4, 128), # channels from last conv * shape from average pooling
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

        # to be used for inference after training
        self.mean = None
        self.std = None

        # self.conv1 = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=filter_size) # 1 input channel, 3 output channels, 5x5 window
        # self.pool = nn.functional.avg_pool2d

        # dummy = ones((1, *frame_shape))

        # dummy = self.conv1(dummy)
        # dummy = self.pool(dummy, kernel_size=self.pooling_kernel_size)
        # print(f"{dummy.shape = }")

        # out_size = dummy.numel()

        # self.fc1 = nn.Linear(out_size, 2)

    def forward(self, x):

        x = self.features(x)
        x = self.regressor(x)
        return x
        # x = self.conv1(x)

        # x = self.pool(x, kernel_size=self.pooling_kernel_size)
        # # flatten for linear
        # x = x.reshape(x.shape[0], -1)

        # x = self.fc1(x)

        # return x

    def predict(self, x):
        # x = (x - mean) / std
        with torch.no_grad():
            x = self.forward(x)
        x = x * self.std + self.mean
        return x

    def save(self):
        torch.save(self.state_dict(), model_dir / "model_weights.pth")
        np.save(model_dir / "model_norm_params.npy", [self.mean, self.std])

    def load(self):
        self.load_state_dict(torch.load(model_dir / "model_weights.pth"))
        mean, std = np.load(model_dir / "model_norm_params.npy")
        self.mean = mean
        self.std = std

class MarkedFrames(torch.utils.data.Dataset): 

    def __init__(self, video_filename, labels_path):
        super().__init__()

        self.data = cv2.VideoCapture(data_dir / video_filename)
        self.data.set(cv2.CAP_PROP_POS_FRAMES, 0)


        self.labels_raw = pandas.read_csv(labels_path)
        self.pipe_mask = mask_creation.get_mask(self.data, "pipe_small")

        self.nonzero = cv2.findNonZero(self.pipe_mask).reshape((-1, 2))
        self.x_min = self.nonzero[:, 0].min()
        self.x_max = self.nonzero[:, 0].max()
        self.y_min = self.nonzero[:, 1].min()
        self.y_max = self.nonzero[:, 1].max()

        # only training data with at least one (should be two) point(s) are valid
        self.available_indices = []
        for key in self.labels_raw.keys():
            if key == "Unnamed: 0":
                continue

            points = self.labels_raw[key]
            valid_points = [point for point in points if isinstance(point, str)]

            # A line fit requires at least two points.
            if len(valid_points) >= 2:
                self.available_indices.append(int(key))

        # for plotting the slope
        self.relevant_xs = np.linspace(30, 50, 100)

        # to optimize data load
        self.previous_index = None

        # labels must be normalized
        self.normalize_labels()


    def normalize_labels(self):
        slopes = []
        biases = []
        for i in range(self.__len__()):
            idx = self.available_indices[i]
            slope, bias = self.get_slope_and_bias(idx)

            slopes.append(slope)
            biases.append(bias)
            # for s, b in zip(slope, bias):
        slopes = np.array(slopes, dtype=np.float32)
        biases = np.array(biases, dtype=np.float32)
        slope_mean = np.mean(slopes)
        bias_mean = np.mean(biases)
        slope_std = np.std(slopes)
        bias_std = np.std(biases)

        self.mean = np.array([slope_mean, bias_mean], dtype=np.float32)
        self.std = np.array([slope_std, bias_std], dtype=np.float32)
        self.labels=np.empty((slopes.shape[0], 2))
        self.labels[:, 0] = (slopes - slope_mean) / slope_std
        self.labels[:, 1] = (biases - bias_mean) / bias_std
        # mean = np.array([-1.2883058, 85.34431], dtype=np.float32)
        # std = np.array([0.42606908, 21.239735], dtype=np.float32)

    def __len__(self):
        return len(self.available_indices)

    def __getitem__(self, apparent_index):
        index = self.available_indices[apparent_index]

        # Seek for the first frame or non-sequential access.
        if self.previous_index is None or index != self.previous_index + 1:
            self.data.set(cv2.CAP_PROP_POS_FRAMES, index)

        self.previous_index = index

        ret, frame = self.data.read()
        if not ret:
            return
        
        frame = frame * self.pipe_mask[:, :, None]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = gray[self.y_min:self.y_max, self.x_min:self.x_max]

        frame = prep_for_torch(frame)

        # slope, bias = self.get_slope_and_bias(index)
        # label = np.array((slope, bias), dtype=np.float32)
        # label = (label - self.mean) / self.std
        # label = tensor(label)
        label = self.labels[apparent_index]

        return frame, label
    
    def get_slope_and_bias(self, index):
        points_str = self.labels_raw[str(index)]

        n_points = len(points_str)
        xs = []
        ys = []
        for i, point in enumerate(points_str):

            if not isinstance(point, str) and np.isnan(point):
                continue
            x, y = point.strip("()").split(",")
            x = int(x)
            y = int(y)
            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.array(ys)
        try:
            slope, bias = np.polyfit(xs, ys, 1)
        except TypeError:
            print(index)
            print(f"{self.labels[str(index)] = }")
            raise TypeError
        slope = slope.astype(np.float32)
        bias = bias.astype(np.float32)

        return slope, bias

    def get_points(self, apparent_index):
        index = self.available_indices[apparent_index]
        points_str = self.labels_raw[str(index)]

        n_points = len(points_str)
        points = []
        for i, point in enumerate(points_str):
            if not isinstance(point, str) and np.isnan(point):
                continue
            x, y = point.strip("()").split(",")
            x = int(x)
            y = int(y)
            points.append(np.array([x, y], dtype=np.uint8))     
        points = np.array(points, dtype=np.uint8)
        return points

class PureVideo:
    def __init__(self, video_filename):
        self.data = cv2.VideoCapture(data_dir / video_filename)
        self.data.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.pipe_mask = mask_creation.get_mask(self.data, "pipe_small")

        self.nonzero = cv2.findNonZero(self.pipe_mask).reshape((-1, 2))
        self.x_min = self.nonzero[:, 0].min()
        self.x_max = self.nonzero[:, 0].max()
        self.y_min = self.nonzero[:, 1].min()
        self.y_max = self.nonzero[:, 1].max()

        self.previous_index = None

    def __getitem__(self, index):
        # Seek for the first frame or non-sequential access.
        if self.previous_index is None or index != self.previous_index + 1:
            self.data.set(cv2.CAP_PROP_POS_FRAMES, index)

        self.previous_index = index

        ret, frame_raw = self.data.read()
        if not ret:
            return
        
        frame = frame_raw * self.pipe_mask[:, :, None]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = gray[self.y_min:self.y_max, self.x_min:self.x_max]

        frame = prep_for_torch(frame)

        return frame, frame_raw

 
def prep_for_torch(frame):
    # convert to interval [0, 1]
    frame = (frame/255).astype(np.float32)

    # add channel dimension
    frame = tensor(frame).unsqueeze(0)

    return frame

def unprep_from_torch(frame):
    # remove channel dimension
    frame = frame.squeeze(0)

    # untorchify
    frame = frame.detach().numpy()

    # convert to [0, 255] and set to unsigned integer
    frame = (frame*255).astype(np.uint8)

    return frame

    
def load_model(load_weights=True):
    model = Linefinder()
    loss_func = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    if load_weights:
        model.load()
        print(f"{model.mean = }")
        print(f"{model.std = }")

    return model, loss_func, optimizer, scheduler




def load_training_data(batch_size):
    data = MarkedFrames(video_filename=video_filename, labels_path=label_dir / "training_labels_v2.csv")

    train_size = int(0.70 * len(data))
    val_size = int(0.15 * len(data))
    test_size = len(data) - train_size - val_size  # Handles rounding errors

    train_data, val_data, test_data = torch.utils.data.random_split(
        data, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return (
        train_data,
        val_data,
        test_data,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        data,
    )


def plot_test(model, data, test_idx):
    model.eval()
    frame, _ = data[test_idx]
    frame = frame.unsqueeze(0) # adding batch dimension

    frame_drawable = unprep_from_torch(frame).squeeze(0) # removing batch dimension

    with torch.no_grad():
        # using predict allows the netork to unnormalize the output
        predictions = model.predict(frame).detach().numpy()
    slope, bias = predictions[0] #* data.target_std + data.target_mean
    xs = np.linspace(0, 50, 100)
    ys = slope * xs + bias
    print(f"{slope = }")
    print(f"{bias = }")
    point1 = (np.uint8(xs[0]),  np.uint8(ys[0]))
    point2 = (np.uint8(xs[-1]), np.uint8(ys[-1]))
    print(f"{point1 = }")
    print(f"{point2 = }")

    cv2.line(frame_drawable, point1, point2, 10000, 1)

    label_points = data.get_points(test_idx)
    for point in label_points:
        print(f"{point = }")
        cv2.drawMarker(frame_drawable, point, 200, 1, markerSize=5)

    cv2.imwrite("figures\\image.png", frame_drawable)


def train(do_train=True, load_weights=True, epochs=2):

    batch_size = 10
    model, loss_func, optimizer, scheduler = load_model(load_weights=load_weights)
    (
        train_data,
        val_data,
        test_data,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        data
    ) = load_training_data(batch_size=batch_size)

    if model.mean is None and model.std is None:
        # training set mean and std are known, we give them directly to the model
        model.mean = data.mean
        model.std = data.std

    if load_weights:
        losses = np.load(model_dir / "losses.npy")
        losses = list(losses)
        validation_losses = np.load(model_dir / "validation_losses.npy")
        validation_losses = list(validation_losses)

    else:
        losses = []
        validation_losses = []

    model.train()
    for epoch in range(epochs):

        if not do_train: 
            break

        model.train()
        # ------------training sequence------------
        for i, batch in enumerate(train_dataloader):
            frames, labels = batch
            optimizer.zero_grad()
            predictions = model(frames)
            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())

        # ------------validation sequence------------
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                frames, labels = batch
                predictions = model(frames)
                validation_loss = loss = loss_func(predictions, labels)
                validation_losses.append(validation_loss.detach().numpy())
                scheduler.step(validation_loss)

                if validation_loss == np.min(validation_losses):
                    # torch.save(model.state_dict(), "model_weights_best.pth")
                    model.save()
        # ------------end of epoch housekeeping------------
        # torch.save(model.state_dict(), "model_weights.pth")
        model.save()
        np.save(model_dir / "losses.npy", losses)
        np.save(model_dir / "validation_losses.npy", validation_losses)


        fig, axs=plt.subplots(1, 2)
        axs[0].plot(losses)
        axs[1].plot(validation_losses)
        axs[0].set_title(f"losses")
        axs[1].set_title(f"validation_losses")
        fig.savefig(figure_dir / "losses.png")
        plt.close()

        print(f"Finshed {epoch+1} / {epochs}")


    # ------------end of traning houesekeeping------------
    losses = np.array(losses)
    validation_losses = np.array(validation_losses)
    np.save(model_dir / "losses.npy", losses)
    np.save(model_dir / "validation_losses.npy", validation_losses)
    model.save()
    # torch.save(model.state_dict(), "model_weights.pth")
    fig, axs=plt.subplots(1, 2)
    axs[0].plot(losses)
    axs[1].plot(validation_losses)
    fig.savefig(figure_dir / "losses.png")
    plt.close()

    # ------------examples of model inference------------
    for i in range(0, len(data), 5):
        plot_test(model, data, i)
        sleep(1)



def livestream_test(start):

    data = PureVideo(video_filename)
    model,_,_,_ = load_model(load_weights=True)
    model.eval()
    show_gray = False

    xs = np.linspace(0, 50, 100)
    i = start
    while data.data.isOpened():
        frame, frame_raw = data[i]
        frame = frame.unsqueeze(0) # add batch dimension

        frame_drawable = unprep_from_torch(frame).squeeze(0)

        with torch.no_grad():
            predictions = model.predict(frame)
        slope, bias = predictions[0].detach().numpy()
        ys = slope * xs + bias
        point1 = (np.uint8(xs[0]),  np.uint8(ys[0]))
        point2 = (np.uint8(xs[-1]), np.uint8(ys[-1]))
        cv2.line(frame_drawable, point1, point2, 10000, 1)
        if show_gray:
            frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Original", frame_raw)
        cv2.imshow("Pipe", frame_drawable)
        # cv2.imwrite("Pipe.png", frame_drawable)
        i += 1
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("s"):
            i = int(input("Go to: "))
        if key == ord(" "):
            new_key = cv2.waitKey(0)
            if new_key == ord("q"):
                break
        if key == ord("g"):
            show_gray = not show_gray

    cv2.destroyAllWindows()


# train(
    # do_train=0,
    # load_weights=1,
    # epochs=1)

# livestream_test(10000)

