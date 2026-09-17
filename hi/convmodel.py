



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

            # -> [B, 16, 24, 26] (half widtha and height)
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

        # to acess training history
        self.training_losses = []
        self.validation_losses = []

    def forward(self, x):

        x = self.features(x)
        x = self.regressor(x)
        return x
 

    def predict(self, x):
        # inference
        with torch.no_grad():
            x = self.forward(x)

        # use the normalization parameters stored from training data
        x = x * self.std + self.mean
        return x

    def save(self):
        """custom save function, ensuring the normalization parameters are saved along with the model"""
        torch.save(self.state_dict(), model_dir / "model_weights.pth")
        np.save(model_dir / "model_norm_params.npy", [self.mean, self.std])
        np.save(model_dir / "training_losses.npy", self.training_losses)
        np.save(model_dir / "validation_losses.npy", self.validation_losses)

    def load(self, best=False):
        """custom load function, ensuring the normalization parameters are loaded along with the model"""
        if best:
            self.load_state_dict(torch.load(model_dir / "model_weights_best.pth"))
        else:
            self.load_state_dict(torch.load(model_dir / "model_weights.pth"))

        self.mean, self.std = np.load(model_dir / "model_norm_params.npy")
        self.training_losses = list(np.load(model_dir / "training_losses.npy"))
        self.validation_losses = list(np.load(model_dir / "validation_losses.npy"))


class MarkedFrames(torch.utils.data.Dataset): 
    """Custom class storing the generated points along with the corresponding video data
    
    TODO: ensure labels and video name are stored together

    Currently only implemented for 
    "training_labels_v2.csv" and "Video_Test_F2787_125743_01_VIDCKPT_sec.mpg"
    """

    def __init__(self, video_filename, labels_path):
        super().__init__()

        # load the video and scroll to start
        self.data = cv2.VideoCapture(data_dir / video_filename)
        self.data.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # load the corresponding labels. TODO: ensure labels and video are connected
        self.labels_raw = pandas.read_csv(labels_path)
        
        # load the mask used to extract pixels at the pipe
        self.pipe_mask = mask_creation.get_mask(self.data, "pipe_small")

        # using the mask, find the rectangle in the frame to feed the model
        self.nonzero = cv2.findNonZero(self.pipe_mask).reshape((-1, 2))
        self.x_min = self.nonzero[:, 0].min()
        self.x_max = self.nonzero[:, 0].max()
        self.y_min = self.nonzero[:, 1].min()
        self.y_max = self.nonzero[:, 1].max()

        # there can be errors in the label set, so we have to sort out the ones with no points
        # only training data with at least two points are valid
        self.available_indices = []
        for key in self.labels_raw.keys():
            # this entry is always present, likely due to initialization of dataframe with no entries
            if key == "Unnamed: 0":
                continue

            # get valid points. They are either strings or nan values. We require at least two string entry
            points = self.labels_raw[key]
            valid_points = [point for point in points if isinstance(point, str)]

            # A line fit requires at least two points.
            if len(valid_points) >= 2:
                self.available_indices.append(int(key))

        # self.available indices now stores the video-indices available from the labeled dataset

        # labels must be normalized
        self.mean = None
        self.std = None
        self.normalize_labels() # internally overwrites the mean and std variables

        # to optimize data load
        self.previous_index = None

        # for plotting the slope
        self.relevant_xs = np.linspace(0, 50, 100)


    def normalize_labels(self):
        """normalize labels using the sample mean and standard deviation
        TODO: rewrite to numpy format
        """
        # get slope and biases using custom loader
        slopes = []
        biases = []
        for i in range(self.__len__()):

            # pick out the frame index, and use it to get the slope and bias
            idx = self.available_indices[i]
            slope, bias = self.get_slope_and_bias(idx)

            slopes.append(slope)
            biases.append(bias)
        slopes = np.array(slopes, dtype=np.float32)
        biases = np.array(biases, dtype=np.float32)

        # initialize labels as slopes and biases, and immediately normalize them
        self.labels=np.empty((slopes.shape[0], 2))
        slope_mean = np.mean(slopes)
        bias_mean = np.mean(biases)
        slope_std = np.std(slopes)
        bias_std = np.std(biases)
        self.labels[:, 0] = (slopes - slope_mean) / slope_std
        self.labels[:, 1] = (biases - bias_mean) / bias_std

        # save the means and stds such that they can be stored in the model later
        self.mean = np.array([slope_mean, bias_mean], dtype=np.float32)
        self.std = np.array([slope_std, bias_std], dtype=np.float32)

    def __len__(self):
        # the available length is the number of labels with valid entries
        return len(self.available_indices)

    def __getitem__(self, apparent_index):
        """



        Setting the frame in a cv2.VideoCapture object is a slow process, and we want to avoid it if possible
        Using self.previous index, we can know whether the frame we read is the next one, and thereby 
        avoid using the set function


        """
        # The torch dataloader uses indicies from 0 to __len___-1, we convert them to the video-indicies here 
        index = self.available_indices[apparent_index]

        # check whether we must use the frame set function, and use it if nessecary
        if self.previous_index is None or index != self.previous_index + 1:
            self.data.set(cv2.CAP_PROP_POS_FRAMES, index)

        self.previous_index = index

        # read data
        ret, frame = self.data.read()
        if not ret:
            return

        # use mask, and convert to grayscale
        frame = frame * self.pipe_mask[:, :, None]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # pick out the pixels in the box surrounding the mask
        frame = gray[self.y_min:self.y_max, self.x_min:self.x_max]

        # convert to interval [0, 1]
        frame = (frame/255).astype(np.float32)

        # add channel dimension
        frame = tensor(frame).unsqueeze(0)

        # pick out the normalized label
        label = self.labels[apparent_index]

        return frame, label
    
    def get_slope_and_bias(self, index):
        """Get the slope and biases from an entry in the label dataset

        requires index to be a video.index, not apparent index
        
        """
        # pick out points from the pandas dataframe
        points_str = self.labels_raw[str(index)]

        # convert points to integers in numpy arrays
        n_points = len(points_str)
        xs = []
        ys = []
        for i, point in enumerate(points_str):
            # ignore the nans, hence the python list implementation
            if not isinstance(point, str) and np.isnan(point):
                continue
            # handle the string formatting
            x, y = point.strip("()").split(",")
            x = int(x)
            y = int(y)
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)

        # use numpy linear regression to get slope and bias (intercept)
        # try-except structure is for debugging
        try:
            slope, bias = np.polyfit(xs, ys, 1)
        except TypeError:
            print(index)
            print(f"{self.labels[str(index)] = }")
            raise TypeError
        
        # convert to flost32 to ensure compatibility with torch
        slope = slope.astype(np.float32)
        bias = bias.astype(np.float32)

        return slope, bias

    def get_points(self, apparent_index):
        """
        not used internally, but provides access to the raw points in the label dataset
        """
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

class MaskedVideo(torch.utils.data.Dataset):
    def __init__(self, video, mask="pipe_small"):
        if isinstance(video, str):
            self.video_filename = video
            self.videocapture = cv2.VideoCapture(data_dir / video_filename)
        elif isinstance(video, cv2.VideoCapture):
            self.videocapture = video
        
        self.videocapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if isinstance(mask, str):
            self.mask = mask_creation.get_mask(self.videocapture, mask)

        elif isinstance(mask, np.ndarray):
            self.mask = mask

        self.nonzero = cv2.findNonZero(self.mask).reshape((-1, 2))
        self.x_min = self.nonzero[:, 0].min()
        self.x_max = self.nonzero[:, 0].max()
        self.y_min = self.nonzero[:, 1].min()
        self.y_max = self.nonzero[:, 1].max()

        self.previous_index = None

    def __len__(self):
        total_frames = int(self.videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames

    def __getitem__(self, index):
        frame_raw = self.get_raw_frame(index)

        frame = frame_raw * self.mask[:, :, None]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = gray[self.y_min:self.y_max, self.x_min:self.x_max]

        frame = prep_for_torch(frame)
        
        return frame

    def get_raw_frame(self, index):
        # Seek for the first frame or non-sequential access.
        if self.previous_index is None or index != self.previous_index + 1:
            self.videocapture.set(cv2.CAP_PROP_POS_FRAMES, index)

        self.previous_index = index

        ret, frame_raw = self.videocapture.read()
        if not ret: 
            return

        return frame_raw

 
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

        model.load(best=True) if load_weights=="best" else model.load()
        print(f"{model.mean = }")
        print(f"{model.std = }")

    return model, loss_func, optimizer, scheduler


def load_data_with_labels()->MarkedFrames:
    data = MarkedFrames(video_filename=video_filename, labels_path=label_dir / "training_labels_v2.csv")
    return data

def prepare_and_split_data(data, batch_size)->tuple:

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
        test_dataloader
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

    cv2.imwrite(figure_dir / "image.png", frame_drawable)


def train(model_entries, data, batch_size=10, epochs=2):
    # unpack the model entries
    model, loss_func, optimizer, scheduler = model_entries

    # prepare data for training
    (
        train_data,
        val_data,
        test_data,
        train_dataloader,
        val_dataloader,
        test_dataloader
    ) = prepare_and_split_data(data, batch_size=batch_size)

    # store the normalization parameters in the model
    if model.mean is None and model.std is None:
        # training set mean and std are known, we give them directly to the model
        model.mean = data.mean
        model.std = data.std


    model.train()
    for epoch in range(epochs):


        model.train()
        # ------------training sequence------------
        for i, batch in enumerate(train_dataloader):
            frames, labels = batch
            optimizer.zero_grad()
            predictions = model(frames)
            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
            model.training_losses.append(loss.detach().numpy())

        # ------------validation sequence------------
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                frames, labels = batch
                predictions = model(frames)
                validation_loss = loss = loss_func(predictions, labels)
                model.validation_losses.append(validation_loss.detach().numpy())
                scheduler.step(validation_loss)

                # in the case where the best validation errors are archived, we save the model parameters
                if validation_loss == np.min(model.validation_losses):
                    torch.save(model.state_dict(), model_dir / "model_weights_best.pth")

        # ------------end of epoch housekeeping------------
        model.save()

        fig, axs=plt.subplots(1, 2)
        axs[0].plot(model.training_losses)
        axs[1].plot(model.validation_losses)
        axs[0].set_title(f"Training losses")
        axs[1].set_title(f"Validation losses")
        fig.savefig(figure_dir / "losses.png")
        plt.close()

        print(f"Finshed {epoch+1} / {epochs}")


    # ------------end of traning houesekeeping------------
    model.save()

    fig, axs=plt.subplots(1, 2)
    axs[0].plot(model.training_losses)
    axs[1].plot(model.validation_losses)
    axs[0].set_title(f"Training losses")
    axs[1].set_title(f"Validation losses")
    fig.savefig(figure_dir / "losses.png")
    plt.close()




def livestream_test(model, start):
    import volumeestimation
    data = MaskedVideo(video_filename)
    model.eval()
    show_gray = False

    frame = data[0]
    # frame.shape = [1, 48, 52] = [C, H, W] -> [H, W]
    frame = frame[0, :, :].numpy()

    pipe_corners = volumeestimation.get_pipe_params(frame)
    
    thickness_history = []

    xs = np.linspace(0, 50, 100)
    i = start
    # set the frame internally in the datastructure
    data[i]
    while data.videocapture.isOpened():
        ret, frame_raw = data.videocapture.read()
        if not ret:
            break

        # cursed manual implementation to ensure MaskedVideo stays useful for torch
        frame = frame_raw * data.mask[:, :, None]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = gray[data.y_min:data.y_max, data.x_min:data.x_max]

        frame = prep_for_torch(frame)

    
        frame = frame.unsqueeze(0) # add batch dimension

        frame_drawable = unprep_from_torch(frame).squeeze(0)

        with torch.no_grad():
            predictions = model.predict(frame)

        slope, bias = predictions[0].detach().numpy()
        ys = slope * xs + bias
        prediction_point1 = (np.int32(xs[0]),  np.int32(ys[0]))
        prediction_point2 = (np.int32(xs[-1]), np.int32(ys[-1]))
        cv2.line(frame_drawable, prediction_point1, prediction_point2, 10000, 1)
        if show_gray:
            frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)

        apparent_pixel_thickness, intersections = volumeestimation.calculate_pixel_apparent_thickness(
            pipe_corners, 
            prediction=predictions[0].detach().numpy()
        )

        for intersection in intersections.values():
            cv2.drawMarker(frame_drawable, intersection, 10000)
        
        thickness_history.append(apparent_pixel_thickness)
        if i%100==0:
            plt.plot(range(start, i+1), thickness_history, c="k")
            plt.xlabel("Index")
            plt.ylabel("Thickness [pixels]")
            plt.title("Apparent pixel thickness")
            plt.savefig(figure_dir / "apparent pixel thickness.png")
            plt.close("all")


        i += 1


        cv2.imshow("Original", frame_raw)
        cv2.imshow("Pipe", frame_drawable)


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

