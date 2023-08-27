import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    """
    A class representing an Inception Block for feature extraction.

    This class implements an Inception Block with multiple branches
    for capturing different time scales in the input data.

    Attributes:
        in_channels (int): Number of input channels.
        out (int): Number of output channels.

    Methods:
        forward(x):
            Forward pass through the InceptionBlock model.
    """

    def __init__(self):
        """
        Initialize the InceptionBlock instance.
        """
        super(InceptionBlock, self).__init__()
        self.in_channels = 1
        self.out = self.in_channels

        # Branch 1: 1x1 Convolution for time scale (sampling = 0.01)
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 Convolution followed by 5x5 Convolution for time scale (sampling = 0.05)
        self.branch7x7 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.out, self.out, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 Convolution followed by 10x10 Convolution for time scale (sampling = 0.10)
        self.branch31x31 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.out, self.out, kernel_size=31, padding=15),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 3x3 Max pooling followed by 1x1 Convolution
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(self.in_channels, self.out, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor) -> tuple:
        """
        Perform a forward pass through the InceptionBlock model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Concatenated output tensor from all branches.
            mean (torch.Tensor): Mean of the input data.
            std (torch.Tensor): Standard deviation of the input data.
        """
        mean = x.mean()
        std = x.std()
        x = torch.divide((x - mean), std)

        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch31x31 = self.branch31x31(x)
        branch_pool = self.branch_pool(x)

        # Concatenate the outputs along the channel dimension
        outputs = [branch1x1, branch7x7, branch31x31, branch_pool]
        return torch.cat(outputs, 1), mean, std
