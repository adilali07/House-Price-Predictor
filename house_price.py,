import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load Data ───────────────────────────────────────────────────────────
df = pd.read_csv('house.csv')
df = df.rename(columns={'Price': 'target'})
df = df.drop(columns=['id', 'Date'], errors='ignore')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nFeatures:\n{list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic Stats:\n{df.describe()}")

# ─── 2. Handle Missing Data ─────────────────────────────────────────────────
df = df.dropna()
print(f"\nShape after dropping nulls: {df.shape}")

# ─── 3. Outlier Removal (IQR on target) ─────────────────────────────────────
Q1 = df['target'].quantile(0.25)
Q3 = df['target'].quantile(0.75)
IQR = Q3 - Q1
before = len(df)
df = df[(df['target'] >= Q1 - 1.5*IQR) & (df['target'] <= Q3 + 1.5*IQR)]
print(f"Removed {before - len(df)} outliers. Shape: {df.shape}")

# ─── 4. Feature Engineering ─────────────────────────────────────────────────
df['RoomsTotal']  = df['number of bedrooms'] + df['number of bathrooms']
df['SqftPerRoom'] = df['living area'] / df['RoomsTotal'].replace(0, 1)
df['HouseAge']    = 2024 - df['Built Year']
df['IsRenovated'] = (df['Renovation Year'] > 0).astype(int)

print("\nEngineered features: RoomsTotal, SqftPerRoom, HouseAge, IsRenovated")

# ─── 5. EDA Plots ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Exploratory Data Analysis', fontsize=14)

# Price distribution
axes[0].hist(df['target'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Price Distribution')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Count')

# Correlation heatmap
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=axes[1])
axes[1].set_title('Correlation Heatmap')

# Price vs Living Area
axes[2].scatter(df['living area'], df['target'], alpha=0.3, s=5, color='steelblue')
axes[2].set_xlabel('Living Area')
axes[2].set_ylabel('Price')
axes[2].set_title('Price vs Living Area')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150)
print("\nEDA plots saved to eda_plots.png")

# ─── 6. Prepare X, y ────────────────────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 7. Models ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_r2  = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
print(f"Linear Regression -> R2: {lr_r2:.4f} | MAE: {lr_mae:.2f} | RMSE: {np.sqrt(lr_mse):.2f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2  = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
print(f"Random Forest     -> R2: {rf_r2:.4f} | MAE: {rf_mae:.2f} | RMSE: {np.sqrt(rf_mse):.2f}")

# ─── 8. Result Plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Results', fontsize=14)

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
feat_imp.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Feature Importance (Random Forest)')

# Predicted vs Actual - LR
axes[1].scatter(y_test, lr_pred, alpha=0.3, s=5, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title(f'Linear Regression (R²={lr_r2:.4f})')
axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')

# Predicted vs Actual - RF
axes[2].scatter(y_test, rf_pred, alpha=0.3, s=5, color='steelblue')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[2].set_title(f'Random Forest (R²={rf_r2:.4f})')
axes[2].set_xlabel('Actual'); axes[2].set_ylabel('Predicted')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150)
print("\nModel plots saved to model_results.png")
