import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

area = np.array([800,1000,1200,1500,1800,2000,2200,2500])
bedrooms = np.array([1,2,2,3,3,4,4,5])
price = np.array([30,38,45,55,65,72,80,92])

X = np.column_stack((area, bedrooms))

X_train, X_test, y_train, y_test = train_test_split(
    X, price, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))
