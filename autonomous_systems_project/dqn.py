import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
        self, h: int, w: int, input_channels: int, hidden_neurons: int, outputs: int
    ):
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
        self.hidden_neurons = hidden_neurons
        self.outputs = outputs

        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # (Size - Kernel size + 2 * Padding) // Stride --> see https://cs231n.github.io/convolutional-networks/
        def conv2d_size_out(size, kernel=5, stride=2) -> int:
            # Here we have no padding --> TODO handle padding
            return (size - kernel) // stride + 1

        # Compute convolution output dimensions
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_height)))

        # Conv output width * conv output height * conv output channels
        self.linear_input_size = convw * convh * 64

        # One hidden linear layer
        self.hidden = nn.Linear(self.linear_input_size, self.hidden_neurons)

        # A fully connected layer for the output
        self.head = nn.Linear(self.hidden_neurons, self.outputs)

    # NN forward pass
    def forward(self, x: torch.Tensor) -> float:
        """
        Forward pass of the network

        Input tensors should have the following shape: (batch_size, image_channels, image_height, image_width)
        """
        # TODO should we use maxpooling? or any other pooling?
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.hidden(x.view(x.size(0), -1)))
        return self.head(x)

    # TODO add best action function
