import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Load data
df = pd.read_csv("AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# === CHAPTER 2: CLEANING AND PREPROCESSING ===
df['Month_num'] = df.index.month
df['Year'] = df.index.year
df['TimeIndex'] = np.arange(len(df))

# Check for nulls and duplicates
print("Null values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Train-test split (80-20)

train = df[:'1959']
test = df['1960':]


# === CHAPTER 3: EDA ===
print("Descriptive Stats:\n", df['#Passengers'].describe())
print("Skewness:", df['#Passengers'].skew())


# Line plot
df['#Passengers'].plot(title='Monthly Passengers Over Time', figsize=(10,5))
plt.xlabel('Date'); plt.ylabel('Passengers'); plt.show()

#Histogram
df['#Passengers'].hist(bins=12, edgecolor='black')
plt.title("Histogram of Passengers"); plt.show()

# Boxplot by Month
sns.boxplot(x='Month_num', y='#Passengers', data=df)
plt.title("Boxplot by Month"); plt.show()

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm"); plt.title("Correlation Matrix"); plt.show()

# Seasonal Decomposition
result = seasonal_decompose(df['#Passengers'], model='multiplicative')
result.plot(); plt.show()


# === CHAPTER 4: MODELING ===

# --- ARIMA ---
model_arima = auto_arima(train['#Passengers'], seasonal=True, m=12, trace=False)
forecast_arima = model_arima.predict(n_periods=len(test))

# Plot ARIMA forecast
plt.plot(train.index, train['#Passengers'], label='Train')
plt.plot(test.index, test['#Passengers'], label='Test')
plt.plot(test.index, forecast_arima, label='ARIMA Forecast', linestyle='--')
plt.title("ARIMA Forecast vs Actual")
plt.legend(); plt.show()


# --- LINEAR REGRESSION ---
lr = LinearRegression()
lr.fit(train['TimeIndex'].values.reshape(-1, 1), train['#Passengers'])
forecast_lr = lr.predict(test['TimeIndex'].values.reshape(-1, 1))

plt.plot(train.index, train['#Passengers'], label='Train')
plt.plot(test.index, test['#Passengers'], label='Test')
plt.plot(test.index, forecast_lr, label='Linear Regression Forecast', linestyle='--')
plt.title("Linear Regression Forecast")
plt.legend(); plt.show()


# --- POLYNOMIAL REGRESSION ---
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(train['TimeIndex'].values.reshape(-1, 1))
lr_poly = LinearRegression()
lr_poly.fit(X_poly, train['#Passengers'])

X_test_poly = poly.transform(test['TimeIndex'].values.reshape(-1, 1))
forecast_poly = lr_poly.predict(X_test_poly)

plt.plot(train.index, train['#Passengers'], label='Train')
plt.plot(test.index, test['#Passengers'], label='Test')
plt.plot(test.index, forecast_poly, label='Poly Regression Forecast', linestyle='--')
plt.title("Polynomial Regression Forecast")
plt.legend(); plt.show()


# === CHAPTER 5: EVALUATION ===

# ARIMA metrics
rmse_arima = np.sqrt(mean_squared_error(test['#Passengers'], forecast_arima))
mape_arima = mean_absolute_percentage_error(test['#Passengers'], forecast_arima)
r2_arima = r2_score(test['#Passengers'], forecast_arima)

# Linear Regression metrics
rmse_lr = np.sqrt(mean_squared_error(test['#Passengers'], forecast_lr))
mape_lr = mean_absolute_percentage_error(test['#Passengers'], forecast_lr)
r2_lr = r2_score(test['#Passengers'], forecast_lr)

# Polynomial Regression metrics
rmse_poly = np.sqrt(mean_squared_error(test['#Passengers'], forecast_poly))
mape_poly = mean_absolute_percentage_error(test['#Passengers'], forecast_poly)
r2_poly = r2_score(test['#Passengers'], forecast_poly)

# Compare results
print("\nModel Evaluation Results:")
print("ARIMA    → RMSE: %.2f | MAPE: %.2f | R²: %.2f" % (rmse_arima, mape_arima, r2_arima))
print("Linear   → RMSE: %.2f | MAPE: %.2f | R²: %.2f" % (rmse_lr, mape_lr, r2_lr))
print("PolyReg  → RMSE: %.2f | MAPE: %.2f | R²: %.2f" % (rmse_poly, mape_poly, r2_poly))
