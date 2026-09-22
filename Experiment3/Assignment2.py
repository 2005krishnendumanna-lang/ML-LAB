import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

data = {
    "Age": [20, 21, None, 23, 24],
    "Salary": [25000, 30000, 28000, None, 40000],
    "Experience": [1, 2, None, 3, 4]
}

df = pd.DataFrame(data)

df[["Age", "Salary", "Experience"]] = SimpleImputer(
    strategy="median"
).fit_transform(df[["Age", "Salary", "Experience"]])

scaler = MinMaxScaler()

df[["Age", "Salary", "Experience"]] = scaler.fit_transform(
    df[["Age", "Salary", "Experience"]]
)

print(df)
