import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

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

for threshold in [0.3, 0.5, 0.7]:
    y_pred = (prob >= threshold).astype(int)

    print("\nThreshold:", threshold)
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
