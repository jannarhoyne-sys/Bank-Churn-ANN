# Bank Customer Churn Prediction (ANN)

A three-layer artificial neural network that predicts whether a bank customer will leave (churn),
using the Churn_Modelling dataset (10,000 customers).

## Approach
- Dropped identifier columns: RowNumber, CustomerId, Surname
- Encoded categorical data: Gender (label encoding), Geography (one-hot encoding)
- 80/20 stratified train/test split (random_state=29), followed by feature scaling
- ANN: 12 inputs → Dense(6, relu) → Dense(6, relu) → Dense(1, sigmoid)
- Compiled with the adam optimizer and binary_crossentropy loss; trained for 100 epochs, batch size 32

## Results (2,000 test customers)
- Accuracy: 84.95%
- Confusion matrix: [[1513, 80], [221, 186]]
- Baseline (predicting everyone stays): 79.65%

## Significant Features
Based on churn rates and effect sizes:
- Significant: Age, IsActiveMember, Geography (Germany), NumOfProducts, Balance, Gender
- Not significant: CreditScore, Tenure, EstimatedSalary, HasCrCard

## Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Files
- `bankchurn_ANNclassifier.py`: Python script
- `bankchurn_ANNclassifier.ipynb`: notebook with outputs
- `Churn_Modelling.csv`: dataset
