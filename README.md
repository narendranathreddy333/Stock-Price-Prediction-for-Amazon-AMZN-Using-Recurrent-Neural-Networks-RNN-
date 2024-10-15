# Stock Price Prediction for Amazon (AMZN) Using Recurrent Neural Networks (RNN)

## Project Overview

This project focuses on predicting the stock price of Amazon (AMZN) using a Recurrent Neural Network (RNN), specifically a Long Short-Term Memory (LSTM) model. By analyzing historical stock price data, the model forecasts future stock prices, demonstrating the potential for time-series forecasting in financial markets.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusions](#conclusions)
- [Technologies Used](#technologies-used)
- [References](#references)

## Dataset

The dataset used for this project was sourced from Kaggle's Nasdaq datasets, which includes:
- Date
- Open Price
- Close Price
- Low Price
- High Price
- Volume
- Adjusted Close

We focus mainly on the 'Close' price for our predictions. The dataset spans from May 15, 1997, to December 12, 2022.

## Model Architecture

The primary model used for prediction is a Long Short-Term Memory (LSTM) neural network. The architecture consists of:
- Input Layer
- LSTM Layers
- Dropout Layers (to prevent overfitting)
- Dense Output Layer

The model's parameters were fine-tuned, including the number of hidden layers, epochs, and learning rates.

## Preprocessing

Key steps in the data preprocessing include:
1. **Data Cleaning**: Handling missing values and outliers.
2. **Normalization**: Scaling the stock prices between 0 and 1 to aid model processing.
3. **Data Splitting**: The dataset was split 80:20 for training and testing purposes.
4. **Reshaping**: The training data was reshaped into sequences of 'n' days to predict the stock price for the next day.

## Evaluation Metrics

The model performance was evaluated using:
- **Root Mean Squared Error (RMSE)**: Measures the difference between predicted and actual stock prices.
- **Mean Squared Error (MSE)**
- **R² Score**: Measures the proportion of the variance in stock prices that the model explains.

## Results

The best performance was achieved using a 6-layer LSTM model with the following parameters:
- **Learning Rate**: 0.1
- **Epochs**: 150
- **RMSE**: 0.022398
- **R² Score**: 0.987

This model demonstrated strong predictive power for the stock price of Amazon.

## Conclusions

Fine-tuning hyperparameters such as the learning rate, number of hidden layers, and number of epochs resulted in improved model performance. The model performs best with relatively stable datasets and may not perform as well when there are sharp fluctuations in the data.

## Technologies Used

- Python
- TensorFlow/Keras
- Pandas, NumPy
- Scikit-learn (for evaluation metrics)
- Matplotlib (for data visualization)
- Kaggle API (for dataset retrieval)

## References

- Y. Kim, "Predicting stock prices with a long short-term memory (LSTM) model," IEEE Access, 2019.
- Q. X. Wu et al., "Stock Price Prediction Using LSTM RNN And CNN-Sliding Window Model," International Conference on AI and Big Data, 2021.
- H. Zhang et al., "Stock price prediction using attention-based multi-input LSTM," IEEE Access, 2019.
