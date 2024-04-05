import pandas as pd

missing_values = ["NA", "na", "-", "n/a"]
df = pd.read_csv("./iris_with_errors.csv", na_values=missing_values)

print(df.isnull().sum())

columns = ["sepal.length", "sepal.width", "petal.length", "petal.width", "variety"]


def replace_outliers_with_median(series):
    median = series[
        (series >= 0) & (series <= 15)
    ].median()  # Calculate median for values within the range
    return series.apply(
        lambda x: median if x < 0 or x > 15 else x
    )  # Use apply with a lambda function


for column in columns[:4]:
    median = df[column].median()
    cnt = 0
    for row in df[column]:
        if row < 0 or row > 15:
            print(row)
            df.loc[cnt, column] = median
        cnt += 1

cnt = 0
correct_values = ["Setosa", "Versicolor", "Virginica"]
for row in df[columns[4]]:
    if row not in correct_values:
        print("Incorrect value: ", row)
        correct_value = -1
        while correct_value not in [1, 2, 3]:
            try:
                correct_value = int(
                    input(
                        """
Correct value with:
1. Setosa
2. Verisicolor
3. Virginica
"""
                    )
                )
            except Exception:
                print("Try again!")
        df.loc[cnt, columns[4]] = correct_value
    cnt += 1
