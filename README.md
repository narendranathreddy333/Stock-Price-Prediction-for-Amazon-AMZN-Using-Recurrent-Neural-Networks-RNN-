# Stock Price Prediction for Amazon (AMZN) Using Recurrent Neural Networks (RNN)


## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

This project aims to predict the stock prices of Amazon (AMZN) using Recurrent Neural Networks (RNN). Stock market forecasting is a complex and challenging task due to the market's highly volatile and non-linear nature. By leveraging the capabilities of RNNs, this project seeks to improve the accuracy of predictions over traditional models.

## Dataset

The dataset used for this project consists of historical stock prices for Amazon, including features such as:

- Open price
- High price
- Low price
- Close price
- Volume traded

Data is preprocessed and split into training and testing sets. The dataset can be found in the `data` folder of this repository.


## Methodology

1. **Data Preprocessing**: 
    - Handling missing values.
    - Normalizing the data for better performance.
    - Splitting the data into train and test sets.

2. **Model Development**:
    - Built an RNN model using LSTM layers.
    - Configured the model with multiple hidden layers to capture the temporal patterns in stock prices.

3. **Model Evaluation**:
    - Evaluated the model's performance using metrics such as Mean Squared Error (MSE).
    - Visualized the actual vs. predicted prices.

## Model Architecture

The model consists of multiple LSTM layers, each with a specific number of neurons and activation functions. Here's a summary of the architecture:

- **Input Layer**: Stock price features
- **LSTM Layer 1**: 50 neurons, ReLU activation
- **LSTM Layer 2**: 50 neurons, ReLU activation
- **Dropout Layer**: 20% dropout rate to prevent overfitting
- **Output Layer**: Single neuron for price prediction


## Results

The model's predictions closely follow the actual stock prices, demonstrating its capability to learn from historical data. Below is a comparison of actual vs. predicted prices.


### Performance Metrics

- **Mean Squared Error (MSE)**: XX.XX
- **Root Mean Squared Error (RMSE)**: XX.XX

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas, Numpy
- Matplotlib for visualization


## Conclusion

This project demonstrates the use of RNNs for predicting stock prices. Although the model achieves satisfactory performance, there's always room for improvement by tuning hyperparameters, using more complex architectures, or incorporating additional features such as market sentiment.

## References

- **Additional Resources**: [TensorFlow Documentation](https://www.tensorflow.org)


