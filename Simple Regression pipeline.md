```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
cars_raw = pd.read_csv('cars_cleaned.csv')
cars_raw.hist(bins=100, legend=True)
print(cars_raw.head())
# Cleaning data
cars_raw['year'] = cars_raw['year'].str.replace('-01-01','')
cars_raw['year'] = cars_raw['year'].astype('int')

cars_m1 = cars_raw.drop([ 'cylinders', 'model','transmission', 'drive', 'size', 'type', 'paint_color'], axis=1, errors=False)
cars_m1.set_index('id')
condition_order = ['other', 'salvage','fair','good','excellent', 'like new', 'new']
condition_mapping = dict(zip(condition_order, [0, 0.2, 0.3, 0.4, 0.6, 0.8, 1]))
cars_m1['condition'] = cars_m1['condition'].map(condition_mapping)

fuel_order = ['other','electric', 'hybrid', 'diesel', 'gas']
fuel_mapping = dict(zip(fuel_order, [0, 0.2, 0.4, 0.8, 1]))
cars_m1['fuel'] = cars_m1['fuel'].map(fuel_mapping)

cars_m1['odometer'] = cars_m1['odometer'].astype('int')

median_per_category = cars_m1.groupby('manufacturer')['price'].median()
normalized_median = (median_per_category - median_per_category.min()) / (median_per_category.max() - median_per_category.min())
cars_m1['manufacturer'] = cars_m1['manufacturer'].map(normalized_median)

cars_m1 = pd.get_dummies(cars_m1, columns=['state'], prefix='state')
print(cars_m1.head())




# Create X y sets
X = cars_m1.drop('price', axis=1)
y = cars_m1.price

# # Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
model = LinearRegression()
model.fit(X_train, y_train)

# # Evaluate model
y_pred = model.predict(X_test)
print(f'MSE: {mean_squared_error(y_test, y_pred)}, R²: {r2_score(y_test, y_pred)}')
```
