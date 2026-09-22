import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

area = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500]).reshape(-1, 1)
price = np.array([30, 38, 45, 55, 65, 72, 80, 92])

X_train, X_test, y_train, y_test = train_test_split(
    area, price, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(area, price)
plt.plot(area, model.predict(area))
plt.xlabel("House Area (sq.ft)")
plt.ylabel("Price (Lakhs)")
plt.show()
