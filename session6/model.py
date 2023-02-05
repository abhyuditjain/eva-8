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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
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
            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            # 30x30x32 | 3x3x32x32 (padding=0) | 28x28x32
            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            # 28x28x32 | 3x3x32x32 (padding=0) | 26x26x32
            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 2
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
        26       9       3           1           1       1       1        11        26
        26      11       3           1           1       1       1        13        26
        26      13       3           1           1       1       1        15        26
        26      15       5           0           2       1       2        19        11
        """
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, dilation=2, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 3
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
        11      19       3           1           1       2       2        23        11
        11      23       3           1           1       2       2        27        11
        11      27       3           1           1       2       2        31        11
        11      31       5           0           1       2       2        35         7
        """
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, dilation=2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 4
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
         7      35       3           1        1         2        2        39        7
         7      39       3           1        1         2        2        43        7
         7      43       3           0        1         2        2        47        3
         3      47       3           0        1         2        2        51        1
        """
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, dilation=2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            # OUTPUT to GAP so not followed by ReLU + BN + Dropout
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, stride=1, padding=0, bias=False),
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
        x = self.gap(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)
