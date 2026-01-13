import  csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(stock):
    with open(stock, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # Skip header as it is column names
        next(csvFileReader)  # Skip second header row
        next(csvFileReader)  # Skip third header row
        for row in csvFileReader:
            try:
                price = float(row[1])  # Close price
            except ValueError:
                continue

            dates.append(len(dates))  # Extract day from date
            prices.append(float(row[1]))
    return

def predict(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel='linear', C=1e3) #liner model
    svr_poly = SVR(kernel='poly', C=1e3, degree=2) #polynomial model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) #radial basis function model; based on euclidean distance

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression for Stock Price Prediction')
    plt.legend()
    plt.show()

    return svr_rbf.predict([[x]])[0], svr_lin.predict([[x]])[0], svr_poly.predict([[x]])[0]

get_data('stock.csv') # CSV file path
predicted_price = predict(dates, prices, 29) # Predict for date 29
print(f"Predicted Prices on day 29:\nRBF Model: {predicted_price[0]}\nLinear Model: {predicted_price[1]}\nPolynomial Model: {predicted_price[2]}") 
