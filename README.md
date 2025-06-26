# Task3-Linear-regression-model
# Importing necessary libraries
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np  # Needed for RMSE

# Load the dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Basic Exploration
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)
print(df[df > 2].count())  # Counts of values greater than 2 for each column
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)
print(df.shape)

# Indexing example
print(df.loc[0:2, "MedInc"])

# Scatterplot visualization
sns.scatterplot(x="MedInc", y="Latitude", color="#ff00ff", data=df)
plt.xlabel("MedInc Data")
plt.ylabel("Latitude Data")
plt.title("MedInc vs Latitude in California")
plt.show()

# Model building
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Correct way to calculate RMSE
r2score = r2_score(y_test, y_pred)

# Printing the Results
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2score)


# Scatter plot with regression line
sns.regplot(x='MedInc', y='MedHouseVal', data=df, scatter_kws={'color': 'purple'}, line_kws={'color': 'green'})

plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Regression Plot: Median Income vs. Median House Value')
plt.show()
