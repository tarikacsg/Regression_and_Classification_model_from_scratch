# Regression_and_Classification_model_from_scratch


This repository contains Python implementations of various operations commonly used in neural networks. These operations include the forward and backward pass for the following layers:

1. Matrix Multiplication Layer:
   - Performs matrix multiplication between the input data and weight matrix.
   - Supports both forward and backward pass calculations.
   
2. Bias Addition Layer:
   - Adds bias terms to the input data.
   - Supports both forward and backward pass calculations.
   
3. Mean Squared Loss Layer:
   - Calculates the mean squared loss between the predicted values and the actual values.
   - Supports both forward and backward pass calculations.

4. Softmax Layer:
   - Applies the softmax activation function to the input data.
   - Supports both forward and backward pass calculations.

5. Sigmoid Layer:
   - Applies the sigmoid activation function to the input data.
   - Supports both forward and backward pass calculations.

6. Cross Entropy Loss Layer:
   - Calculates the cross entropy loss between the predicted probabilities and the true labels.
   - Supports both forward and backward pass calculations.

## Regression Model Training

In addition to the implementation of these operations, this repository provides an example of training a regression model using the **Boston Housing Pricing dataset** from scikit-learn. The implementation includes a stochastic gradient descent (SGD) function for training the model.

### Matrix Multiplication Layer--> Bias Addition Layer--> MSE Layer

To train the regression model, follow these steps:

1. Load the Boston Housing Pricing dataset using the `sklearn.datasets.load_boston()` function.
2. Preprocess the dataset as needed (e.g., normalization, splitting into training and testing sets).
3. Initialize the necessary neural network layers and set their parameters.
4. Use the implemented SGD function to train the model.
5. Evaluate the trained model using appropriate metrics and visualize the results.

## Multi-Class Classifier

Furthermore, this repository demonstrates the creation of a multi-class classifier using the **Iris dataset** from scikit-learn. 

### Matrix Multiplication Layer--> Bias Addition Layer--> Softmax Layer--> Cross Entropy Layer

To create the multi-class classifier, follow these steps:

1. Load the Iris dataset using the `sklearn.datasets.load_iris()` function.
2. Preprocess the dataset as needed (e.g., normalization, splitting into training and testing sets).
3. Initialize the necessary neural network layers and set their parameters.
4. Train the classifier using the implemented SGD function.
5. Evaluate the trained classifier using appropriate metrics and visualize the results.



