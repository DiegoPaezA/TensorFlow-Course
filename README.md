# TensorFlow-Course

This is my code repository for the Course [TensorFlow Developer Certificate in 2023: Zero to Mastery
](https://is.gd/qG2T02), published by Daniel Bourke.

## 0 — [TensorFlow Fundamentals](https://is.gd/fBwXms)

- Introduction to tensors (creating tensors)
- Getting information from tensors (tensor attributes)
- Manipulating tensors (tensor operations)
- Tensors and NumPy
- Using @tf.function (a way to speed up your regular Python functions)
- Using GPUs with TensorFlow

## 1 — [Neural Network Regression with TensorFlow](https://is.gd/Z5C7xc)

- Architecture of a regression model
- Input shapes and output shapes
  - X: features/data (inputs)
  - y: labels (outputs)
- Creating custom data to view and fit
- Steps in modelling
  - Creating a model
  - Compiling a model
        - Defining a loss function
        - Setting up an optimizer
        - Creating evaluation metrics
  - Fitting a model (getting it to find patterns in our data)
- Evaluating a model
  - Visualizng the model ("visualize, visualize, visualize")
  - Looking at training curves
  - Compare predictions to ground truth (using our evaluation metrics)
- Saving a model (so we can use it later)
- Loading a model

## 2 — [Neural Network Classification with TensorFlow](https://tinyurl.com/2h7686m4)

- Architecture of a classification model
- Input shapes and output shapes
  - `X`: features/data (inputs)
  - `y`: labels (outputs) 
    - "What class do the inputs belong to?"
- Creating custom data to view and fit
- Steps in modelling for binary and mutliclass classification
  - Creating a model
  - Compiling a model
    - Defining a loss function
    - Setting up an optimizer
      - Finding the best learning rate
    - Creating evaluation metrics
  - Fitting a model (getting it to find patterns in our data)
  - Improving a model
- The power of non-linearity
- Evaluating classification models
  - Visualizng the model ("visualize, visualize, visualize")
  - Looking at training curves
  - Compare predictions to ground truth (using our evaluation metrics)