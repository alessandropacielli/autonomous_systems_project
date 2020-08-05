import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, h: int, w: int, input_channels: int, outputs: int):
        super(DQN, self).__init__()

        """
    Construct a new DQN object.
    
    :param h: The height of the image.
    :param w: The width of the image.
    :param input_channels: The number of channels of the image (e.g. 3 for RGB images)
    :param outputs: The number of outputs.
    """

        self.input_channels = input_channels
        self.input_height = h
        self.input_width = w
        self.outputs = outputs

        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # (Size - Kernel size + 2 * Padding) // Stride --> see https://cs231n.github.io/convolutional-networks/
        def conv2d_size_out(size, kernel=5, stride=2):
            return (size - kernel) // stride + 1

        # Compute convolution output dimensions
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_height)))

        # Conv output width * conv output height * conv output channels
        self.linear_input_size = convw * convh * 64

        # A fully connected layer for the output
        self.head = nn.Linear(self.linear_input_size, self.outputs)

    # NN forward pass
    def forward(self, x: torch.Tensor):
        """
    Forward pass of the network
    """
        # TODO should we use maxpooling? or any other pooling?
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    # TODO add best action function
