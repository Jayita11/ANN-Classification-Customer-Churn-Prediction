# ANN-Classification-Customer-Churn-Prediction

This project aims to predict customer churn using an Artificial Neural Network (ANN) model. The project includes data preprocessing, model training, and deployment using a Streamlit app.



## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Collection](#data-collection)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Data Preprocessing](#data-preprocessing)
  - [Splitting Data](#splitting-data)
  - [Model Building](#model-building)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)
  - [Mixed Data Types](#mixed-data-types)
  - [Class Imbalance](#class-imbalance)
  - [Model Optimization](#model-optimization)
  - [Prediction Interpretation](#prediction-interpretation)
- [Installation](#installation)
- [Usage](#usage)
  - [Experiments](#experiments)
  - [Model Prediction](#model-prediction)
  - [Deployment](#deployment)
- [License](#license)

## Overview

This code develops an artificial neural network model to perform binary classification on a bank customer churn dataset to predict whether a customer will leave the bank or not. It loads and preprocesses the dataset, does one-hot encoding of categorical variables, and develops and trains a simple multi-layer perceptron model using TensorFlow Keras with two hidden layers. The model is compiled and trained on the preprocessed training set. Then its predictive performance is evaluated on the held-out test set by generating predictions and calculating classification metrics like a confusion matrix and accuracy score. The code thus demonstrates the basic end-to-end workflow of developing, training, and evaluating an artificial neural network classifier on a real-world classification problem involving preprocessing of categorical variables.

## Dataset

The dataset used in this project is `Churn_Modelling.csv`. It contains customer information and churn status, which is used to train the ANN model to predict churn.

## Methodology

### Data Collection

Gather the data required for the project. The specified dataset contains information about customers, including features like credit score, geography, gender, age, tenure, balance, and more.


### Exploratory Data Analysis

Explore and analyze the dataset to understand its characteristics. This involves:
- Checking for missing values
- Handling duplicates
- Visualizing distributions of variables
- Exploring relationships between variables

The EDA process helps to gain insights into the data and make informed decisions about preprocessing steps.

### Data Preprocessing

Clean and prepare the data for modeling. Steps may include:
- Removing null values
- Dropping unnecessary columns (e.g., RowNumber, CustomerId, Surname)
- Handling duplicate records
- Label encoding categorical variables
- Standardizing numerical features

### Splitting Data

Divide the dataset into training and testing sets. The training set is used to train the ANN, while the testing set is used to evaluate its performance.

### Model Building

Construct an Artificial Neural Network for predicting customer churn. Design the architecture of the neural network, including the number of layers, activation functions, and neurons. A sequential model with layers of dense neurons is used.

### Model Training

Train the ANN using the training set. Adjust model hyperparameters such as the number of epochs, batch size, and learning rate. Monitor the training process using TensorBoard for performance visualization. The trained model is saved as model.h5.

#### TensorBoard Visualization

Monitor the training process using TensorBoard for performance visualization. TensorBoard helps in visualizing the following metrics:
- **Accuracy**: Track the accuracy of the model over epochs.
- **Loss**: Observe how the loss decreases over training epochs.
- **Learning Rate**: Visualize how the learning rate changes during training.

To set up TensorBoard, follow these steps:

1. Import TensorBoard and create a callback:
    ```python
    from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
    import datetime

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    ```

2. Include the callback in the model training process:
    ```python
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        callbacks=[tensorflow_callback,early_stopping_callback]
    )
    ```

3. To launch TensorBoard, use the following command in your terminal:
    ```sh
    %tensorboard --logdir=logs/fit
    ```

The trained model is saved as model.h5.

![unnamed-45](https://github.com/user-attachments/assets/a920526b-51a8-4717-be68-79cde0adbe13)
![unnamed-46](https://github.com/user-attachments/assets/3e45467d-a0d0-48a2-a27e-9219f04d87da)
![unnamed-47](https://github.com/user-attachments/assets/c4beb2a6-800c-4677-908d-8995d14f6f73)



### Model Evaluation

Evaluate the trained model using the testing set. Calculate metrics such as accuracy, precision, recall, and F1-score. Generate a confusion matrix and a classification report to assess the model's performance.

![unnamed-48](https://github.com/user-attachments/assets/f8c66064-ffc0-4033-a7db-52f332854128)
![unnamed-49](https://github.com/user-attachments/assets/f510b0c3-74a4-4692-8bb1-7e4eb65177ad)


### Model Prediction

Use the prediction.ipynb notebook to demonstrate how to use the trained model for predicting customer churn on new data. Load the pre-trained model and apply it to new datasets to generate predictions.

## Results

Achieved an impressive accuracy of 85.9% with the developed Artificial Neural Network (ANN) model, showcasing the effectiveness of the predictive model in customer churn prediction.

### Deployment
Deploy the model using a Streamlit app (app.py). The app allows users to input customer data and get churn predictions. To run the app, execute the following command:

![Animation_1](https://github.com/user-attachments/assets/b1aa898c-50bb-42c0-8d4d-22183a596efa)



This starts a web server and opens the app in the default web browser, enabling interaction with the model for churn predictions.


## Challenges and Solutions

### Mixed Data Types

- Label encoding was used to convert categorical 'Gender' to numeric.
- One-hot encoding handled multi-class 'Geography'.
- No text features in this dataset. Normalization handled other numeric types.

### Class Imbalance

- Models generally perform poorly on imbalanced classes. Over-sampling could be used to duplicate minority class examples.
- Using accuracy alone as a metric would mask poor minority class prediction. Confusion matrix helps identify true/false positives and negatives.

### Model Optimization

- Started with a simple 2 hidden layer network, gradually adjusted the number of units, activations, dropouts, etc.
- Tried additional convolutional/LSTM layers since sequence/images were unavailable.
- Used callback functions like EarlyStopping to prevent overfitting.
- Permutation feature importance helped identify impactful predictors.

### Prediction Interpretation

- Studied relationships between features and targets via visualization.
- Identified customer profiles most/least likely to churn based on predictions.
- Used model to simulate retention programs - if changes are made profile is unlikely to churn.

## Installation

To run this project, you need to have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository:
    ```sh
    git clone https://github.com/Jayita11/ANN-Classification-Customer-Churn-Prediction/tree/main.git
    cd ANN-Classification-Customer-Churn-Prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

