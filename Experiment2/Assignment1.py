import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df.corr())

sns.scatterplot(data=df, x="alcohol", y="malic_acid")
plt.show()

sns.histplot(df["alcohol"], kde=True)
plt.show()
