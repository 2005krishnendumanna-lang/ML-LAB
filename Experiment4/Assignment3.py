import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

X = np.array([800,1000,1200,1500,1800,2000,2200,2500]).reshape(-1,1)
y = np.array([30,38,45,55,65,72,80,92])

linear = LinearRegression()
linear.fit(X, y)
linear_pred = linear.predict(X)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
poly_pred = model.predict(X_poly)

print("Linear R2 Score:", r2_score(y, linear_pred))
print("Polynomial R2 Score:", r2_score(y, poly_pred))
