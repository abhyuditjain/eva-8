# Part 1

In part 1 of the assignment, we were given a network and we were supposed to find all the partial derivatives of loss with respect to the weights. We were also supposed to plot the loss for some number of iterations for various learning rates.

## Network

![Alt text](./images/network.png "Network")

The above image shows a network with the following inputs.

```
η = Learning rate
t1 = output1 (Expected output for Input 1)
t2 = output2 (Expected output for Input 2)
i1 = Input 1
i2 = Input 2
```

Some other notations used:

```
w1 = weight 1
w2 = weight 2
w3 = weight 3
w4 = weight 4
w5 = weight 5
w6 = weight 6
w7 = weight 7
w8 = weight 8

h1 = hidden output 1
h2 = hidden output 2
a_h1 = activated hidden output 1
a_h2 = activated hidden output 2
o1 = output 1
o2 = output 2
a_o1 = activated output 1
a_o2 = activated output 2
```

Also, the loss function used is the L2 function, defined as:

```
E1 = (1/2) * (t1 - a_o1)^2
E2 = (1/2) * (t2 - a_o2)^2
```

Based on the above, we calculate the following parameters. Every parameter except the weights is initially undefined. Initial weights are chosen randomly here and these are updated in future iterations based on the gradient. Other parameters, including weights for future iterations, are derived from other parameters and input.

```
h1 = w1*i1 + w2*i2
h2 = w3*i1 + w4*i2

a_h1 = σ(h1) = 1/(1 + exp(-h1))
a_h2 = σ(h2) = 1/(1 + exp(-h2))

o1 = w5*a_h1 + w6*a_h2
o2 = w7*a_h1 + w8*a_h2

a_o1 = σ(o1) = 1/(1 + exp(-o1))
a_o2 = σ(o2) = 1/(1 + exp(-o2))

E1 = (1/2)*(t1 - a_o1)^2
E2 = (1/2)*(t2 - a_o2)^2

E_Total = E1 + E2 = E
```

Now to update weights, we need the partial differential of `E` with respect to those weights.

In short, we need the following:

```
∂E/∂w1 = ?
∂E/∂w2 = ?
∂E/∂w3 = ?
∂E/∂w4 = ?
∂E/∂w5 = ?
∂E/∂w6 = ?
∂E/∂w7 = ?
∂E/∂w8 = ?
```

Let us start from the right to left (backward) direction.

```
∂E/∂w5 = ∂(E1 + E1)/∂w5 = ∂E1/∂w5           [as E2 does not depend on w5]
∂E/∂w6 = ∂(E1 + E1)/∂w6 = ∂E1/∂w6           [as E2 does not depend on w6]
∂E/∂w7 = ∂(E1 + E1)/∂w7 = ∂E2/∂w7           [as E1 does not depend on w7]
∂E/∂w8 = ∂(E1 + E1)/∂w8 = ∂E2/∂w8           [as E1 does not depend on w8]
```

We can derive the above by chaining the differentials like:

```
∂E/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5               (i)
∂E/∂w6 = ∂E1/∂w6 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w6               (ii)
∂E/∂w7 = ∂E2/∂w7 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂w7               (iii)
∂E/∂w8 = ∂E2/∂w8 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂w8               (iv)
```

To derive `(i)`, we calculate the 3 terms in it:

```
∂E1/∂a_o1 = ∂((1/2)*(t1 - a_o1)^2)/∂a_o1 = a_o1 - t1

∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = σ(o1) * (1 - σ(o1)) = a_o1 * (1 - a_o1)

∂o1/∂w5 = ∂(w5*a_h1 + w6*a_h2)/∂w5 = a_h1
```

Plugging the above in `(i)`, we get:

```
∂E/∂w5 = (a_o1 - t1) * (a_o1 * (1 - a_o1)) * a_h1
```

Similarly, to derive `(ii)`, we calculate the 3 terms in it:

```
∂E1/∂a_o1 = ∂((1/2)*(t1 - a_o1)^2)/∂a_o1 = a_o1 - t1

∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = σ(o1) * (1 - σ(o1)) = a_o1 * (1 - a_o1)

∂o1/∂w6 = ∂(w5*a_h1 + w6*a_h2)/∂w6 = a_h2
```

Plugging the above in `(ii)`, we get:

```
∂E/∂w6 = (a_o1 - t1) * (a_o1 * (1 - a_o1)) * a_h2
```

Similarly, to derive `(iii)`, we calculate the 3 terms in it:

```
∂E2/∂a_o2 = ∂((1/2)*(t2 - a_o2)^2)/∂a_o1 = a_o2 - t2

∂a_o2/∂o2 = ∂(σ(o2))/∂o2 = σ(o2) * (1 - σ(o2)) = a_o2 * (1 - a_o2)

∂o2/∂w7 = ∂(w7*a_h1 + w8*a_h2)/∂w7 = a_h1
```

Plugging the above in `(iii)`, we get:

```
∂E/∂w7 = (a_o2 - t2) * (a_o2 * (1 - a_o2)) * a_h1
```

Similarly, to derive `(iv)`, we calculate the 3 terms in it:

```
∂E2/∂a_o2 = ∂((1/2)*(t2 - a_o2)^2)/∂a_o1 = a_o2 - t2

∂a_o2/∂o2 = ∂(σ(o2))/∂o2 = σ(o2) * (1 - σ(o2)) = a_o2 * (1 - a_o2)

∂o2/∂w8 = ∂(w7*a_h1 + w8*a_h2)/∂w8 = a_h2
```

Plugging the above in `(iv)`, we get:

```
∂E/∂w8 = (a_o2 - t2) * (a_o2 * (1 - a_o2)) * a_h2
```

Finally, we have our 4 of the 8 derivatives:

```
∂E/∂w5 = (a_o1 - t1) * (a_o1 * (1 - a_o1)) * a_h1
∂E/∂w6 = (a_o1 - t1) * (a_o1 * (1 - a_o1)) * a_h2
∂E/∂w7 = (a_o2 - t2) * (a_o2 * (1 - a_o2)) * a_h1
∂E/∂w8 = (a_o2 - t2) * (a_o2 * (1 - a_o2)) * a_h2
```

Similarly, we can go ahead and derive the other 4.

```
∂E/∂w1 = ∂(E1 + E2)/∂w1 = ∂E1/∂w1 + ∂E2/∂w1                     (v)
∂E/∂w2 = ∂(E1 + E2)/∂w2 = ∂E1/∂w2 + ∂E2/∂w2                     (vi)
∂E/∂w3 = ∂(E1 + E2)/∂w3 = ∂E1/∂w3 + ∂E2/∂w3                     (vii)
∂E/∂w4 = ∂(E1 + E2)/∂w4 = ∂E1/∂w4 + ∂E2/∂w4                     (viii)
```

To derive the above, we need to find the derivatives of the errors with respect to `a_h1` and `a_h2` because

```
∂E1/∂w1 = ∂E1/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
∂E2/∂w1 = ∂E2/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
∂E1/∂w2 = ∂E1/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2
∂E2/∂w2 = ∂E2/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2
∂E1/∂w3 = ∂E1/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3
∂E2/∂w3 = ∂E2/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3
∂E1/∂w4 = ∂E1/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w4
∂E2/∂w4 = ∂E2/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w4

∂E1/∂a_h1 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1
          = (a_o1 - t1) * (a_o1 * (1 - a_o1)) * w5

∂E2/∂a_h1 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂a_h1
          = (a_o2 - t2) * (a_o2 * (1 - a_o2)) * w7

∂E1/∂a_h2 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h2
          = (a_o1 - t1) * (a_o1 * (1 - a_o1)) * w6

∂E2/∂a_h2 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂a_h2
          = (a_o2 - t2) * (a_o2 * (1 - a_o2)) * w8
```

Using the above equations, we can derive `∂E/∂a_h1` and `∂E/∂a_h2` (simple addition):

```
∂E/∂a_h1 = ∂E1/∂a_ah1 + ∂E2/∂a_h1
         = ((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w5) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w7)

∂E/∂a_h2 = ∂E1/∂a_ah2 + ∂E2/∂a_h2
         = ((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w6) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w8)
```

Now we can finally get the desired derivatives with respect to the weights.

```
∂E/∂w1 = ∂E/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
       = [((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w5) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w7)] * a_h1 * (1 - a_h1) * i1

∂E/∂w2 = ∂E/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2
       = [((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w5) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w7)] * a_h1 * (1 - a_h1) * i2

∂E/∂w3 = ∂E/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3
       = [((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w6) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w8)] * a_h2 * (1 - a_h2) * i1

∂E/∂w4 = ∂E/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w4
       = [((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w6) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w8)] * a_h2 * (1 - a_h2) * i2
```

The final equations of derivative of error with respect to weights are:

```
∂E/∂w1 = [((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w5) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w7)] * a_h1 * (1 - a_h1) * i1

∂E/∂w2 = [((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w5) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w7)] * a_h1 * (1 - a_h1) * i2

∂E/∂w3 = [((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w6) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w8)] * a_h2 * (1 - a_h2) * i1

∂E/∂w4 = [((a_o1 - t1) * (a_o1 * (1 - a_o1)) * w6) + ((a_o2 - t2) * (a_o2 * (1 - a_o2)) * w8)] * a_h2 * (1 - a_h2) * i2

∂E/∂w5 = (a_o1 - t1) * (a_o1 * (1 - a_o1)) * a_h1

∂E/∂w6 = (a_o1 - t1) * (a_o1 * (1 - a_o1)) * a_h2

∂E/∂w7 = (a_o2 - t2) * (a_o2 * (1 - a_o2)) * a_h1

∂E/∂w8 = (a_o2 - t2) * (a_o2 * (1 - a_o2)) * a_h2
```

Now, we know the gradient. All that's left to do is update the individual weights. For this, we multiply learning rate with the gradient with respect to the weight and subtract the result from that weight.

So, new weights would be calculated as:

```
w1 = w1 - η * ∂E/∂w1
w2 = w2 - η * ∂E/∂w2
w3 = w3 - η * ∂E/∂w3
w4 = w4 - η * ∂E/∂w4
w5 = w5 - η * ∂E/∂w5
w6 = w6 - η * ∂E/∂w6
w7 = w7 - η * ∂E/∂w7
w8 = w8 - η * ∂E/∂w8
```

We keep doing this to reduce the total error or loss.

See the screenshot of the spreadsheet using the above formulae (`η = 1`).

![Alt text](./images/spreadsheet_eta_1.png "Spreadsheet")

## Plots

### η = 0.1

![Alt text](./images/spreadsheet_eta_0.1_plot.png "Plot 0.1")

### η = 0.2

![Alt text](./images/spreadsheet_eta_0.2_plot.png "Plot 0.2")

### η = 0.5

![Alt text](./images/spreadsheet_eta_0.5_plot.png "Plot 0.5")

### η = 0.8

![Alt text](./images/spreadsheet_eta_0.8_plot.png "Plot 0.8")

### η = 1.0

![Alt text](./images/spreadsheet_eta_1.0_plot.png "Plot 1.0")

### η = 2.0

![Alt text](./images/spreadsheet_eta_2.0_plot.png "Plot 2.0")

<br>
<br>
<br>

# Part 2

In part 2 of the assignment, we had to refer to the following [Network](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx), which identifies MNIST images, and achieve the following:

- [x] 99.4% validation accuracy `Achieved in Epochs - 13, 16, 17`
- [x] Less than 20k Parameters `Total Parameters = 17,962`
- [x] Less than 20 Epochs `19 Epochs`
- [x] Usage of Batch Normalization, Dropout, Fully Connected (FC) Layer, Global Average Pooling (GAP) `Refer to the architecture`

<br>

## Architecture

```
LAYER            [INPUT    | KERNELS    | OUTPUT    | Receptive field]

CONV1            [28x28x1  | 5x5x1x32   | 24x24x32  | RF = 5x5]
ReLU
BATCH_NORM(32)

CONV2            [24x24x32 | 3x3x32x16  | 22x22x16  | RF = 7x7]
ReLU
BATCH_NORM(16)
MAX_POOLING      [22x22x16 |  k=2, s=2  | 11x11x16  | RF = 14x14]
DROPOUT(0.1)

CONV3            [11x11x16 | 3x3x16x16  | 9x9x16    | RF = 16x16]
ReLU
BATCH_NORM(16)

CONV4            [9x9x16   | 3x3x16x64  | 7x7x64    | RF = 18x18]
ReLU
BATCH_NORM(64)

GAP(1)
DROPOUT(0.1)

FC               [64*1*1 => 10]
```

## Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 24, 24]             832
       BatchNorm2d-2           [-1, 32, 24, 24]              64
            Conv2d-3           [-1, 16, 22, 22]           4,624
       BatchNorm2d-4           [-1, 16, 22, 22]              32
           Dropout-5           [-1, 16, 11, 11]               0
            Conv2d-6             [-1, 16, 9, 9]           2,320
       BatchNorm2d-7             [-1, 16, 9, 9]              32
            Conv2d-8             [-1, 64, 7, 7]           9,280
       BatchNorm2d-9             [-1, 64, 7, 7]             128
AdaptiveAvgPool2d-10             [-1, 64, 1, 1]               0
          Dropout-11             [-1, 64, 1, 1]               0
           Linear-12                   [-1, 10]             650
================================================================
Total params: 17,962
Trainable params: 17,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.48
Params size (MB): 0.07
Estimated Total Size (MB): 0.55
----------------------------------------------------------------
```

<br>

## Link to the solution

[<font size=10><u><b>Solution</b></u></font>](https://colab.research.google.com/drive/1UCCuRF1rhUaQ6RZ5B5_shFS7PeGb6z_Q?usp=sharing)

Or you can check out the file: `S4_Assignment_Solution.ipynb`
<br>

## Logs

```
Epoch:  1
loss=0.04502683877944946 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.44it/s]

Test set: Average loss: 0.0772, Accuracy: 9779/10000 (97.79%)

Epoch:  2
loss=0.09046034514904022 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.80it/s]

Test set: Average loss: 0.0548, Accuracy: 9827/10000 (98.27%)

Epoch:  3
loss=0.019108165055513382 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.76it/s]

Test set: Average loss: 0.0402, Accuracy: 9875/10000 (98.75%)

Epoch:  4
loss=0.029182815924286842 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.34it/s]

Test set: Average loss: 0.0335, Accuracy: 9904/10000 (99.04%)

Epoch:  5
loss=0.008654656819999218 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.10it/s]

Test set: Average loss: 0.0271, Accuracy: 9922/10000 (99.22%)

Epoch:  6
loss=0.0032439972274005413 batch_id=937: 100%|██████████| 938/938 [00:20<00:00, 46.43it/s]

Test set: Average loss: 0.0277, Accuracy: 9907/10000 (99.07%)

Epoch:  7
loss=0.049506206065416336 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.03it/s]

Test set: Average loss: 0.0243, Accuracy: 9921/10000 (99.21%)

Epoch:  8
loss=0.1322145313024521 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.67it/s]

Test set: Average loss: 0.0249, Accuracy: 9929/10000 (99.29%)

Epoch:  9
loss=0.0013750848593190312 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.64it/s]

Test set: Average loss: 0.0247, Accuracy: 9923/10000 (99.23%)

Epoch:  10
loss=0.14347663521766663 batch_id=937: 100%|██████████| 938/938 [00:18<00:00, 49.59it/s]

Test set: Average loss: 0.0232, Accuracy: 9925/10000 (99.25%)

Epoch:  11
loss=0.00669552106410265 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.38it/s]

Test set: Average loss: 0.0202, Accuracy: 9938/10000 (99.38%)

Epoch:  12
loss=0.010275788605213165 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 49.25it/s]

Test set: Average loss: 0.0221, Accuracy: 9932/10000 (99.32%)

Epoch:  13
loss=0.012313640676438808 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 49.02it/s]

Test set: Average loss: 0.0189, Accuracy: 9948/10000 (99.48%)

Epoch:  14
loss=0.019488289952278137 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.48it/s]

Test set: Average loss: 0.0225, Accuracy: 9936/10000 (99.36%)

Epoch:  15
loss=0.0010803210316225886 batch_id=937: 100%|██████████| 938/938 [00:18<00:00, 49.75it/s]

Test set: Average loss: 0.0204, Accuracy: 9938/10000 (99.38%)

Epoch:  16
loss=0.0032050954177975655 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 47.77it/s]

Test set: Average loss: 0.0198, Accuracy: 9947/10000 (99.47%)

Epoch:  17
loss=0.0024224459193646908 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.16it/s]

Test set: Average loss: 0.0203, Accuracy: 9940/10000 (99.40%)

Epoch:  18
loss=0.011659398674964905 batch_id=937: 100%|██████████| 938/938 [00:20<00:00, 46.67it/s]

Test set: Average loss: 0.0184, Accuracy: 9936/10000 (99.36%)

Epoch:  19
loss=0.017939575016498566 batch_id=937: 100%|██████████| 938/938 [00:19<00:00, 48.72it/s]

Test set: Average loss: 0.0180, Accuracy: 9939/10000 (99.39%)
```
