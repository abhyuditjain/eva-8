import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        """
        nout = (n_in + 2p - k)/s + 1
        jout = jin * s
        rout = rin + (k - 1)*jin

        INPUT BLOCK
        NIn    RFIn   KernelSize  Padding    Stride    JumpIn  JumpOut   RFOut   NOut
        32      1       3           1           1        1        1       3       32
        """
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 1
        NIn    RFIn   KernelSize  Padding    Stride  JumpIn  JumpOut   RFOut     NOut
        32      3       3           0           1       1       1        5        30
        30      5       3           0           1       1       1        7        28
        28      7       3           0           1       1       1        9        26
        """
        self.conv_block1 = nn.Sequential(
            # 32x32x32 | 3x3x32x32 (padding=0) | 30x30x32
            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            # 30x30x32 | 3x3x32x32 (padding=0) | 28x28x32
            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            # 28x28x32 | 3x3x32x32 (padding=0) | 26x26x32
            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 2
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
        26       9       3           1           1       1       1        11        26
        26      11       3           1           1       1       1        13        26
        26      13       3           1           1       1       1        15        26
        26      15       5           0           1       1       1        19        22
        """
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, dilation=2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 3
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
        22      19       3           0           1       1       1        21        20
        20      21       3           0           1       1       1        23        18
        18      23       3           0           1       1       1        25        16
        16      25       5           0           1       1       1        29        12
        """
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, dilation=2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 4
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
        12      29       3           0           1       1       1        31        10
        10      31       3           0           1       1       1        33         8
         8      33       3           0           1       1       1        37         4
        """
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, dilation=2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        OUTPUT BLOCK
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
         4      37       3           0           1       1       1        39         2
        """
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, stride=1, padding=0, bias=False),
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.output(x)
        x = self.gap(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)
