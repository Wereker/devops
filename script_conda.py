import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import csv

with open("bottle.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    sal_idx = header.index("Salnty")
    temp_idx = header.index("T_degC")

data = np.genfromtxt("bottle.csv", delimiter=",", skip_header=1, dtype=float)

sal = data[:, sal_idx]
temp = data[:, temp_idx]

mask = ~np.isnan(sal) & ~np.isnan(temp)
sal = sal[mask]
temp = temp[mask]

X = sal.reshape(-1, 1)
y = temp.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
score = regr.score(X_test, y_test)
print("R^2 score:", score)

plt.scatter(sal, temp, alpha=0.3, label="Data points")
x_line = np.linspace(sal.min(), sal.max(), 100).reshape(-1, 1)
y_line = regr.predict(x_line)
plt.plot(x_line, y_line, color="red", linewidth=2, label="Linear Regression")

plt.xlabel("Sal (Salnty)")
plt.ylabel("Temp (T_degC)")
plt.title("Relationship between Salinity and Temperature")
plt.savefig("plot_conda.png")