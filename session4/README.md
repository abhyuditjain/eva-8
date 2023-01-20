# Objective

The objective of the assignment was to train a model to achieve the following with an MNIST dataset:

- More than 99.40% testing accuracy at multiple epochs in the end
- Less than 10,000 parameters or if possible, less than 8,000 parameters
- Less than or equal to 15 Epochs
- Show Receptive Field calculations
- Use at least 3 iterations

## Note

My receptive field calculations are in a table in a text block in the individual `.ipynb` files.

<hr>

# Iteration 1

This was just a setup of various helpers for loading data, training, testing etc.
The model was quite heavy.

### Target:

- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop

### Results:

- Parameters = 6,379,786
- Best training accuracy = 99.95%
- Best testing accuracy = 99.23%

### Analysis:

- Extremely Heavy Model for such a problem
- Model is over-fitting
- This was just to setup the structure of the code and we'll improve upon it in later iterations

[Solution](https://colab.research.google.com/drive/1_rXYQQliGodDmC3vV-TBHRV0sV2iYb-h?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_01.ipynb`
<br>

<hr>

# Iteration 2

## Target:

- We want to get the skeleton structure right so that we can add minimum changes to future iterations at a time

## Results:

- Parameters = 194,884
- Best training accuracy = 99.25%
- Best testing accuracy = 98.84%

## Analysis:

- Model is large, our target parameters = 10,000
- Model is over-fitting

[Solution](https://colab.research.google.com/drive/1krrKyCZcCVPJFcIMYOf_xAXgukzY3Bc7?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_02.ipynb`
<br>

<hr>

# Iteration 3

## Target:

- To make the model lighter i.e. less number of parameters

## Results:

- Parameters = 10,790
- Best training accuracy = 98.88%
- Best testing accuracy = 98.60%

## Analysis:

- Model is still larger than our target, but it's still a good model
- Model is not over-fitting as evident by the delta in training and testing accuracy. If training accuracy is improved, I can see the model achieving the target testing accuracy

[Solution](https://colab.research.google.com/drive/13j6uf6xDEDrM26oCkdJMDuJASrn8UvPs?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_03.ipynb`
<br>

<hr>

# Iteration 4

## Target:

- To make the model with acceptable number of parameters i.e. <= 10,000

## Results:

- Parameters = 9,990
- Best training accuracy = 99.06%
- Best testing accuracy = 98.82%

## Analysis:

- Model has achieved its target size
- Model is not over-fitting as evident by the delta in training and testing accuracy. If training accuracy is improved, I can see the model achieving the target testing accuracy

[Solution](https://colab.research.google.com/drive/1twPuBueHjDyjgPHSnb03M_lbKeXwclKB?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_04.ipynb`
<br>

<hr>

# Iteration 5

## Target:

- Add batch-normalization to increase model efficiency.

## Results:

- Parameters = 10,154
- Best training accuracy = 99.82%
- Best testing accuracy = 99.20%

## Analysis:

- Model has exceeded its target size after adding batch normalization
- Model is overfitting as evident by accuracies from epoch 10 onwards, the delta is increasing

[Solution](https://colab.research.google.com/drive/1RakRJwBtd3Y8f1b2jR8AsBZtrjjExnEc?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_05.ipynb`
<br>

<hr>

# Iteration 6

## Target:

- Add regularization (dropout) to make learning harder in hopes of increasing testing accuracy

## Results:

- Parameters = 10,154
- Best training accuracy = 99.35%
- Best testing accuracy = 99.30%

## Analysis:

- Model has exceeded its target size after adding batch normalization in previous iteration (batch normalization). Parameters were unaffected (as expected) after adding dropout
- Model is not over-fitting, but I could not achieve the target testing accuracy

[Solution](https://colab.research.google.com/drive/1il9V9YUgI3miHsZdI9StFc7W2THq17EQ?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_06.ipynb`
<br>

<hr>

# Iteration 7

## Target:

- Add GAP and remove the last convolution with big kernel.

## Results:

- Parameters = 5,254
- Best training accuracy = 98.56%
- Best testing accuracy = 98.98%

## Analysis:

- Model is well within target size, in fact, it has achieved the secondary target size (< 8000)
- Model is not over-fitting, but the accuracy has dropped in comparison to the previous iteration, which is expected as the number of parameters has also dropped significantly.
- In the next iteration, might need to increase the number of parameters to fairly compare the accuracies.

[Solution](https://colab.research.google.com/drive/1WAo3FCUf4PUVEwyiKQSI2Ae7CFKIBBz9?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_07.ipynb`
<br>

<hr>

# Iteration 8

## Target:

- Add more parameters to make this model comparable to a similarly sized model (before GAP)
- Max pooling after RF = 5 as it is MNIST data where features start to form at that field

## Results:

- Parameters = 13,808
- Best training accuracy = 99.37%
- Best testing accuracy = 99.46%

## Analysis:

- Model exceeded target size
- Model is not over-fitting.
- We reached our target testing accuracy

  - 99.42% at Epoch 12
  - 99.46% at Epoch 13

- Our target is not achieved consistently (during later epochs)
- I will play with the training data now to make the training harder

[Solution](https://colab.research.google.com/drive/1TvlmxjRw_kz1nT9yLGi50a1B1UMTLCV-?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_08.ipynb`
<br>

<hr>

# Iteration 9

## Target:

- Added a random rotation to our training data to make the training even harder

## Results:

- Parameters = 13,808
- Best training accuracy = 99.26%
- Best testing accuracy = 99.50%

## Analysis:

- Model exceeded target size
- Model is under-fitting as expected.
- We reached our target testing accuracy

  - 99.50% at Epoch 10
  - 99.49% at Epoch 14

- Overall testing accuracy is up and we achieved our target during the last epoch. And accuracy overall is high during the later epochs.
- Since accuracy is going up and down, I'll play with learning rate in the next iteration

[Solution](https://colab.research.google.com/drive/1h-jI8Q1CDtIKuicyiAYvfK3VGR9VxtSr?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_09.ipynb`
<br>

<hr>

# Iteration 10

## Target:

- Added StepLR Scheduler to change learning rate

## Results:

- Parameters = 13,808
- Best training accuracy = 99.24%
- Best testing accuracy = 99.44%

## Analysis:

- Model exceeded target size
- Model is under-fitting as expected.
- We reached our target testing accuracy faster and more consistently
  - 99.42% at Epoch 6
  - 99.40% at Epoch 7
  - 99.40% at Epoch 8
  - 99.41% at Epoch 9
  - 99.44% at Epoch 10
  - 99.42% at Epoch 11
  - 99.40% at Epoch 12
  - 99.43% at Epoch 13
  - 99.41% at Epoch 14
- Overall testing accuracy is up and we achieved our target during the last 9 epochs. And accuracy overall is high during the later epochs.
- Will try playing with lower parameters

[Solution](https://colab.research.google.com/drive/1C-lD1i8ysVACtpixM0-c7oEXozRdgb2I?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_10.ipynb`
<br>

<hr>

# Iteration 11

## Target:

- Changed the initial LR and steps for LR scheduler
  - Initial LR = 0.3
  - steps = 5
  - gamma = 0.1
- Reduced the number of parameters from 13,808 to 7,750

## Results:

- Parameters = 7,750
- Best training accuracy = 99.19%
- Best testing accuracy = 99.46%

## Analysis:

- Model well within target size, even less than 8,000
- Model is under-fitting as expected.
- We reached our target testing accuracy faster and more consistently
  - 99.40% at Epoch 6
  - 99.46% at Epoch 7
  - 99.44% at Epoch 8
  - 99.42% at Epoch 9
  - 99.43% at Epoch 10
  - 99.42% at Epoch 11
  - 99.44% at Epoch 12
  - 99.42% at Epoch 14
- Overall testing accuracy is up and we achieved our target during the last 7 epochs. And accuracy overall is high during the later epochs.
- Epoch 13 had an accuracy of 99.39%

[Solution](https://colab.research.google.com/drive/1zVbNIKCOccxqhkUa96jTOUCmkR2cR2bW?usp=sharing)

Or you can check out the file: `session4_assignment_iteration_11.ipynb`
<br>

<hr>
