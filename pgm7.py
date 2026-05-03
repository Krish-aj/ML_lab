# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Function for Linear Regression using California Housing Dataset
def linear_regression_california():

    # Load dataset
    housing = fetch_california_housing(as_frame=True)

    # Selecting feature and target
    X = housing.data[["AveRooms"]]
    y = housing.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Plot graph
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")

    plt.xlabel("Average number of rooms (AveRooms)")
    plt.ylabel("Median value of homes ($100,000)")
    plt.title("Linear Regression - California Housing Dataset")
    plt.legend()
    plt.show()

    # Evaluation
    print("Linear Regression - California Housing Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))


# Function for Polynomial Regression using Auto MPG Dataset
def polynomial_regression_auto_mpg():

    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

    column_names = ["mpg", "cylinders", "displacement", "horsepower",
                    "weight", "acceleration", "model_year", "origin"]

    # Load dataset
    data = pd.read_csv(url, sep='\s+', names=column_names, na_values="?")

    # Remove missing values
    data = data.dropna()

    # Feature and target
    X = data["displacement"].values.reshape(-1, 1)
    y = data["mpg"].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Polynomial Regression Model
    poly_model = make_pipeline(
        PolynomialFeatures(degree=2),
        StandardScaler(),
        LinearRegression()
    )

    # Train model
    poly_model.fit(X_train, y_train)

    # Prediction
    y_pred = poly_model.predict(X_test)

    # Plot graph
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, y_pred, color="red", label="Predicted")

    plt.xlabel("Displacement")
    plt.ylabel("Miles per gallon (mpg)")
    plt.title("Polynomial Regression - Auto MPG Dataset")
    plt.legend()
    plt.show()

    # Evaluation
    print("\nPolynomial Regression - Auto MPG Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))


# Main function
if __name__ == "__main__":

    print("Demonstrating Linear Regression and Polynomial Regression\n")

    linear_regression_california()

    polynomial_regression_auto_mpg()
