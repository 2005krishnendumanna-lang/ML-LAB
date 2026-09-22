import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 1, 2],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 40, 45],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, threshold = roc_curve(y_test, prob)
auc = roc_auc_score(y_test, prob)

print("AUC:", auc)

plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
