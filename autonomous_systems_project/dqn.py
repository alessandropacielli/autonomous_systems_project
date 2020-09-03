import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariDQN(nn.Module):
    def __init__(
        self, frame_h: int, frame_w: int, input_channels: int, num_actions: int
    ):
        """
        Construct a new DQN object for Atari games.
        The networks has 3 convolutional layers:
            * The first one convolves 32 6*6 kernels with stride 3
            * The second one convolves 64 4*4 kernels with stride 2
            * The last one convolves 64 4*4 kernels with stride 1
        
        Parameters:
            frame_h: The height of the input frame in pixels.
            frame_w: The width of the input frame in pixels.
            input_channels: The number of channels of the image (e.g. 3 for RGB images, 4 for b&w frame-stacked environments)
            num_actions: The number of possible actions (outputs of the net).
        """
        super(AtariDQN, self).__init__()

        self.input_shape = (input_channels, frame_h, frame_w)

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=6, stride=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # (Size - Kernel size + 2 * Padding) // Stride --> see https://cs231n.github.io/convolutional-networks/
        def conv2d_size_out(size, kernel=5, stride=2) -> int:
            return (size - kernel) // stride + 1

        # Compute number of inputs for dense layer
        convh = conv2d_size_out(frame_h, kernel=6, stride=3)
        convh = conv2d_size_out(convh, kernel=4, stride=2)
        convh = conv2d_size_out(convh, kernel=4, stride=1)

        convw = conv2d_size_out(frame_w, kernel=6, stride=3)
        convw = conv2d_size_out(convw, kernel=4, stride=2)
        convw = conv2d_size_out(convw, kernel=4, stride=1)

        linear_input = convh * convw * 64

        # Dense layer and output
        self.fc1 = nn.Linear(linear_input, 512)
        self.head = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.fc1(x.view(x.size(0), -1)))
        x = self.head(x) * 15
        return x

    def input_shape(self):
        return self.input_shape


class SimpleDQN(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        """
        Constructs a DQN for problems with 1-dimensional inputs.

        """
        super(SimpleDQN, self).__init__()

        self.input_shape = num_inputs

        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.head = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.head(x))
        return x

    def input_shape(self):
        return self.input_shape
