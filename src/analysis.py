import pandas as pd

# Allow show data details

def showDatasetDetails(df):
    print("First 5 rows of the DataFrame:")
    print(df.head())

    print("\nDataFrame Information:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nShape of DataFrame:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())


def unique_values_for_each_column(df):
    columns = df.columns.tolist()
    columns.remove("id")
    columns.remove("productDisplayName")
    for col in columns:
        print(df[col].value_counts())
        print("------------------------------")