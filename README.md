# House Price Predictor

A machine learning project to predict house prices using Linear Regression and Random Forest.

## Results
| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.70 | ~121,971 |
| Random Forest | 0.87 | ~68,982 |

## Dataset
[House Price Prediction Dataset](https://www.kaggle.com/code/souravchanda01/house-price-dataset-of-india/input) - 14,000+ samples, 20+ features

## Features Engineered
- `RoomsTotal` - total bedrooms + bathrooms
- `SqftPerRoom` - living area per room
- `HouseAge` - age of house from built year
- `IsRenovated` - binary flag for renovation

## Project Structure
```
HomeLens/
├── code.py       # main script
├── house.csv            # dataset
└── README.md
```
