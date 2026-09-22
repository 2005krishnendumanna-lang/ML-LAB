import numpy as np
import pandas as pd

data = {
    "Name":["Krishnendu", "Aryan", "Haladhar", "Anubhav", "Mohak"],
    "Roll_No":[1, 2, 3, 4, 5],
    "Marks":[80, 85, 65, 90, 84],
    "Attendance":[88, 92, 68, 72, 81]
}

df = pd.DataFrame(data)

def grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    else:
        return "F"

df["Grade"] = df["Marks"].apply(grade)

print(df)
