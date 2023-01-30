import torch.nn as nn
import torch.nn.functional as F


def normalizer(method, out_channels):
    if method not in ["BN", "GN", "LN"]:
        raise ValueError("Invalid method of normalization")

    if method == "BN":
        return nn.BatchNorm2d(out_channels)
    elif method == "LN":
        return nn.GroupNorm(1, out_channels)
    else:
        return nn.GroupNorm(4, out_channels)


class Net(nn.Module):
    def __init__(self, normalization_method="BN"):
        """
        Default normalization = batch normalization
        """
        super(Net, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 26
            nn.ReLU(),
            normalizer(normalization_method, 10),
            nn.Dropout(0.05),
        )

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 24
            nn.ReLU(),
            normalizer(normalization_method, 20),
            nn.Dropout(0.05),
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # output_size = 24
            nn.MaxPool2d(2, 2),  # output_size = 12
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 10
            nn.ReLU(),
            normalizer(normalization_method, 20),
            nn.Dropout(0.05),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 8
            nn.ReLU(),
            normalizer(normalization_method, 10),
            nn.Dropout(0.05),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 6
            nn.ReLU(),
            normalizer(normalization_method, 10),
            nn.Dropout(0.05),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # output_size = 6
            nn.ReLU(),
            normalizer(normalization_method, 10),
            nn.Dropout(0.05),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # output_size = 1

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.conv6(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)
