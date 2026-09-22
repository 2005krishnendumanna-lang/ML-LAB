from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = [[20], [21], [22], [23], [24]]

print("StandardScaler:")
print(StandardScaler().fit_transform(data))

print("\nMinMaxScaler:")
print(MinMaxScaler().fit_transform(data))
