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

        CONVOLUTION BLOCK 1
        NIn    RFIn   KernelSize  Padding    Stride  JumpIn  JumpOut   RFOut     NOut
        32      1       3           1           1       1       1        3        32
        32      3       3           1           1       1       1        5        32
        32      5       3           1           1       1       1        7        32
        32      7       3           1           1       1       1        9        32
        """
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        nout = (n_in + 2p - k)/s + 1
        jout = jin * s
        rout = rin + (k - 1)*jin

        CONVOLUTION BLOCK 2
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
        32       9      3+2          1         2        1        2        13        15
        15      13       3           1         1        2        2        17        15
        15      17       3           1         1        2        2        21        15
        15      21       3           1         1        2        2        25        15
        """
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        nout = (n_in + 2p - k)/s + 1
        jout = jin * s
        rout = rin + (k - 1)*jin

        CONVOLUTION BLOCK 3
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
        15      25      3+2          1         2         2       4        33         7
         7      33       3           1         1         4       4        41         7
         7      41       3           1         1         4       4        49         7
         7      49       3           1         1         4       4        57         7
        """
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        nout = (n_in + 2p - k)/s + 1
        jout = jin * s
        rout = rin + (k - 1)*jin

        CONVOLUTION BLOCK 4
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut
         7      57      3+2          1        2         4        8        73        3
         3      73       3           1        1         8        8        89        3
         3      89       3           1        1         8        8        105       3
         3      105      3           0        1         8        8        121       1
        """
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            # OUTPUT to GAP so not followed by ReLU + BN + Dropout
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, stride=1, padding=0, bias=False),
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=1),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.gap(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)
