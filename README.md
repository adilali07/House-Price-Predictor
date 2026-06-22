# HomeLens - House Price Predictor

A machine learning project to predict house prices using Linear Regression and Random Forest.

## Results
| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.70 | ~121,971 |
| Random Forest | 0.87 | ~68,982 |

## Dataset
[House Price Prediction Dataset](https://www.kaggle.com/) - 14,000+ samples, 20+ features

## Features Engineered
- `RoomsTotal` - total bedrooms + bathrooms
- `SqftPerRoom` - living area per room
- `HouseAge` - age of house from built year
- `IsRenovated` - binary flag for renovation

## Setup
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python house_price.py
```

## Project Structure
```
HomeLens/
├── house_price.py       # main script
├── house.csv            # dataset
├── eda_plots.png        # EDA visualizations
├── model_results.png    # model result plots
└── README.md
```
