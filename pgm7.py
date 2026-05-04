import pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
housing = fetch_california_housing(as_frame=True)
X, y = housing.data[["AveRooms"]], housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test, y_test); plt.plot(X_test, y_pred, 'r')
plt.title("Linear Regression"); plt.show()

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))


# Polynomial Regression
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin"]

data = pd.read_csv(url, sep='\s+', names=column_names, na_values="?").dropna()

X = data["displacement"].values.reshape(-1,1)
y = data["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly_model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression())
poly_model.fit(X_train, y_train)

y_pred = poly_model.predict(X_test)

plt.scatter(X_test, y_test); plt.scatter(X_test, y_pred, c='r')
plt.title("Polynomial Regression"); plt.show()

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
